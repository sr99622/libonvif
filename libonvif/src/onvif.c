/*******************************************************************************
* onvif.c
*
* copyright 2018, 2023, 2024 Stephen Rhodes
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*******************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include "libxml/xpathInternals.h"

#ifdef _WIN32
    #include <ws2tcpip.h>
    #include <winsock2.h>
    #include <wincrypt.h>
    #include <iphlpapi.h>
    #include <fcntl.h>
	#pragma comment(lib, "iphlpapi.lib")
	#pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <ifaddrs.h>
    #include <sys/ioctl.h>
    #include <sys/types.h>
    #include <netdb.h>
    #include <net/if.h>
    #include <netinet/in.h>
    #include <sys/time.h>
#endif

#include "onvif.h"
#include "sha1.h"
#include "cencode.h"
#include <stdint.h>
#include <errno.h>

#define INT_TO_ADDR(_addr) \
(_addr & 0xFF), \
(_addr >> 8 & 0xFF), \
(_addr >> 16 & 0xFF), \
(_addr >> 24 & 0xFF)

xmlDocPtr sendCommandToCamera(char * cmd, char * xaddrs);
void getBase64(unsigned char * buffer, int chunk_size, unsigned char * result);
void getUUID(char uuid_buf[47]);
void addUsernameDigestHeader(xmlNodePtr root, xmlNsPtr ns_env, char * user, char * password, time_t offset);
void addHttpHeader(xmlDocPtr doc, xmlNodePtr root, char * xaddrs, char * post_type, char cmd[], int cmd_length);
int checkForXmlErrorMsg(xmlDocPtr doc, char error_msg[1024]);
int getXmlValue(xmlDocPtr doc, xmlChar *xpath, char buf[], int buf_length);
int getNodeAttributen (xmlDocPtr doc, xmlChar *xpath, xmlChar *attribute, char buf[], int buf_length, int profileIndex);
#define getNodeAttribute(doc,xpath,attribute,buf,buf_length) getNodeAttributen(doc,xpath,attribute,buf,buf_length,0)
xmlXPathObjectPtr getNodeSet (xmlDocPtr doc, xmlChar *xpath);


const int SHA1_DIGEST_SIZE = 20;
char preferred_network_address[16];
static bool dump_reply = false;
static void dumpReply(xmlDocPtr reply);

int getNetworkInterfaces(struct OnvifData *onvif_data) {
    memset(onvif_data->ip_address_buf, 0, sizeof(onvif_data->ip_address_buf));
    memset(onvif_data->networkInterfaceToken, 0, sizeof(onvif_data->networkInterfaceToken));
    memset(onvif_data->networkInterfaceName, 0, sizeof(onvif_data->networkInterfaceName));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetNetworkInterfaces", NULL);
    char cmd[4096] = {'/0'};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body//tds:GetNetworkInterfacesResponse//tds:NetworkInterfaces";
        xmlNodeSetPtr nodeset;
        xmlChar *enabled = NULL;
        xmlXPathObjectPtr xml_result = getNodeSet(reply, xpath);
        xmlDocPtr temp_doc = xmlNewDoc(BAD_CAST "1.0");
        if (xml_result) {
            nodeset = xml_result->nodesetval;
            for (int i=0; i<nodeset->nodeNr; i++) {
                xmlNodePtr cur = nodeset->nodeTab[i];
                xmlChar *token = xmlGetProp(cur, BAD_CAST "token");
                xmlDocSetRootElement(temp_doc, cur);

                bool dhcp = false;
                char isDHCP[128] = {'/0'};
                xpath = BAD_CAST "//tds:NetworkInterfaces//tt:IPv4//tt:Config//tt:DHCP";
                if (getXmlValue(temp_doc, xpath, isDHCP, 128) == 0) {
                    if (strcmp(isDHCP, "true") == 0) {
                        dhcp = true;
                    }
                    onvif_data->dhcp_enabled = dhcp;
                }

                xmlChar *xpath_address;
                xmlChar *xpath_prefix;
                if (dhcp) {
                    xpath_address = BAD_CAST "//tds:NetworkInterfaces//tt:IPv4//tt:Config//tt:FromDHCP//tt:Address";
                    xpath_prefix = BAD_CAST "//tds:NetworkInterfaces//tt:IPv4//tt:Config//tt:FromDHCP//tt:PrefixLength";
                } else {
                    xpath_address = BAD_CAST "//tds:NetworkInterfaces//tt:IPv4//tt:Config//tt:Manual//tt:Address";
                    xpath_prefix = BAD_CAST "//tds:NetworkInterfaces//tt:IPv4//tt:Config//tt:Manual//tt:PrefixLength";
                }

                char ip_address_buf[128] = {'/0'};
                if (getXmlValue(temp_doc, xpath_address, ip_address_buf, 128) == 0) {
                    char host[128] = {'/0'};
                    extractHost(onvif_data->xaddrs, host);

                    if (strcmp(ip_address_buf, host) == 0) {
                        strcpy(onvif_data->ip_address_buf, ip_address_buf);
                        strcpy(onvif_data->networkInterfaceToken, (char*) token);
                        char prefix_length_buf[128];
                        if (getXmlValue(temp_doc, xpath_prefix, prefix_length_buf, 128) ==  0) {
                            onvif_data->prefix_length = atoi(prefix_length_buf);
                        }
                        xpath = BAD_CAST "//tds:NetworkInterfaces//tt:Info//tt:Name";
                        getXmlValue(temp_doc, xpath, onvif_data->networkInterfaceName, 128);
                        i = nodeset->nodeNr;
                    }
                }
    
                xmlFree(token);
            }
            xmlXPathFreeObject(xml_result);
        }
        xmlFreeDoc(temp_doc);
        if (enabled != NULL) {
            xmlFree(enabled);
        }

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getNetworkInterfaces");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getNetworkInterfaces - No XML reply");
    }
    return result;
    return 0;
}

int setNetworkInterfaces(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setNetworkInterfaces = xmlNewTextChild(body, ns_tds, BAD_CAST "SetNetworkInterfaces", NULL);
    xmlNewTextChild(setNetworkInterfaces, ns_tt, BAD_CAST "InterfaceToken", BAD_CAST onvif_data->networkInterfaceName);
    xmlNodePtr networkInterface = xmlNewTextChild(setNetworkInterfaces, ns_tt, BAD_CAST "NetworkInterface", NULL);
    xmlNodePtr ipv4 = xmlNewTextChild(networkInterface, ns_tt, BAD_CAST "IPv4", NULL);
    if (onvif_data->dhcp_enabled) {
        xmlNewTextChild(ipv4, ns_tt, BAD_CAST "DHCP", BAD_CAST "true");
    } else {
        xmlNewTextChild(ipv4, ns_tt, BAD_CAST "DHCP", BAD_CAST "false");
        xmlNodePtr manual = xmlNewTextChild(ipv4, ns_tt, BAD_CAST "Manual", NULL);
        xmlNewTextChild(manual, ns_tt, BAD_CAST "Address" , BAD_CAST onvif_data->ip_address_buf);
        char prefix_length_buf[128];
        sprintf(prefix_length_buf, "%d", onvif_data->prefix_length);
        xmlNewTextChild(manual, ns_tt, BAD_CAST "PrefixLength", BAD_CAST prefix_length_buf);
    }
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body//tds:SetNetworkInterfacesResponse//tds:RebootNeeded";
        char rebootNeeded[128];
        if (getXmlValue(reply, xpath, rebootNeeded, 128) == 0) {
            if (strcmp(rebootNeeded, "true") == 0) {
                rebootCamera(onvif_data);
            }
        }

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setNetworkInterfaces");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setNetworkInterfaces - No XML reply");
    }
    return result;
}

int getNetworkDefaultGateway(struct OnvifData *onvif_data) {
    memset(onvif_data->default_gateway_buf, 0, sizeof(onvif_data->default_gateway_buf));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetNetworkDefaultGateway", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body//tds:GetNetworkDefaultGatewayResponse//tds:NetworkGateway//tt:IPv4Address";
        getXmlValue(reply, xpath, onvif_data->default_gateway_buf, 128);
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getNetworkDefaultGateway");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getNetworkDefaultGateway - No XML reply");
    }
    return result;
}

int setNetworkDefaultGateway(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setNetworkDefaultGateway = xmlNewTextChild(body, ns_tds, BAD_CAST "SetNetworkDefaultGateway", NULL);
    xmlNewTextChild(setNetworkDefaultGateway, ns_tt, BAD_CAST "IPv4Address", BAD_CAST onvif_data->default_gateway_buf);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setNetworkDefaultGateway");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setNetworkDefaultGateway - No XML reply");
    }
    return result;
}

int getDNS(struct OnvifData *onvif_data) {
    memset(onvif_data->dns_buf, 0, sizeof(onvif_data->dns_buf));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetDNS", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body//tds:GetDNSResponse//tds:DNSInformation//tt:FromDHCP";
        char fromDHCP[128];
        if (getXmlValue(reply, xpath, fromDHCP, 128) == 0) {
            if (strcmp(fromDHCP, "true") == 0) {
                xpath = BAD_CAST "//s:Body//tds:GetDNSResponse//tds:DNSInformation//tt:DNSFromDHCP//tt:IPv4Address";
                if (getXmlValue(reply, xpath, onvif_data->dns_buf, 128) == 0) {}
            } else {
                xpath = BAD_CAST "//s:Body//tds:GetDNSResponse//tds:DNSInformation//tt:DNSManual//tt:IPv4Address";
                if (getXmlValue(reply, xpath, onvif_data->dns_buf, 128) == 0) {}
            }
        }
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getDNS");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getDNS - No XML reply");
    }
    return result;
}

int setDNS(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    char fromDHCP_buf[128];
    if (onvif_data->dhcp_enabled) {
        strcpy(fromDHCP_buf, "true");
    } else {
        strcpy(fromDHCP_buf, "false");
    }

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);        if (result < 0)
            strcat(onvif_data->last_error, " setDNS");

    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setDNS = xmlNewTextChild(body, ns_tds, BAD_CAST "SetDNS", NULL);
    if (!onvif_data->dhcp_enabled) {
        xmlNodePtr dnsManual = xmlNewTextChild(setDNS, ns_tds, BAD_CAST "DNSManual", NULL);
        xmlNewTextChild(dnsManual, ns_tt, BAD_CAST "Type", BAD_CAST "IPv4");
        xmlNewTextChild(dnsManual, ns_tt, BAD_CAST "IPv4Address", BAD_CAST onvif_data->dns_buf);
    } else {
        xmlNewTextChild(setDNS, ns_tds, BAD_CAST "FromDHCP", BAD_CAST fromDHCP_buf);
    }
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setDNS");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setDNS - No XML reply");
    }
    return result;
}

int getNTP(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetNTP", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
		xmlChar *xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:FromDHCP";
		char ntp_buf[128];
		getXmlValue(reply, xpath, ntp_buf, 128);
		if (strcmp(ntp_buf,"true") == 0) {
			onvif_data->ntp_dhcp = true;
			xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPFromDHCP//tt:Type";
			getXmlValue(reply, xpath, onvif_data->ntp_type, 128);
			if (strcmp(onvif_data->ntp_type,"IPv4") == 0)
				xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPFromDHCP//tt:IPv4Address";
			else if (strcmp(onvif_data->ntp_type,"IPv4") == 0)
				xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPFromDHCP//tt:IPv6Address";
			else
				xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPFromDHCP//tt:DNSname";
			getXmlValue(reply, xpath, onvif_data->ntp_addr, 128);
		} else {
			onvif_data->ntp_dhcp = false;
			xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPManual//tt:Type";
			getXmlValue(reply, xpath, onvif_data->ntp_type, 128);
			if (strcmp(onvif_data->ntp_type,"IPv4") == 0)
				xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPManual//tt:IPv4Address";
			else if (strcmp(onvif_data->ntp_type,"IPv4") == 0)
				xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPManual//tt:IPv6Address";
			else
				xpath = BAD_CAST "//s:Body//tds:GetNTPResponse//tt:NTPManual//tt:DNSname";
			getXmlValue(reply, xpath, onvif_data->ntp_addr, 128);
		}
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getNTP");
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getNTP - No XML reply");
    }
    return result;
}

int setNTP(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    char fromDHCP_buf[128];
    if (onvif_data->ntp_dhcp) {
        strcpy(fromDHCP_buf, "true");
    } else {
        strcpy(fromDHCP_buf, "false");
    }
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setNTP = xmlNewTextChild(body, ns_tds, BAD_CAST "SetNTP", NULL);

    xmlNewTextChild(setNTP, ns_tds, BAD_CAST "FromDHCP", BAD_CAST fromDHCP_buf);
    if (!onvif_data->ntp_dhcp) {
        xmlNodePtr ntpManual = xmlNewTextChild(setNTP, ns_tds, BAD_CAST "NTPManual", NULL);
        xmlNewTextChild(ntpManual, ns_tt, BAD_CAST "Type", BAD_CAST onvif_data->ntp_type);
		if (strcmp(onvif_data->ntp_type,"IPv4") == 0)
			xmlNewTextChild(ntpManual, ns_tt, BAD_CAST "IPv4Address", BAD_CAST onvif_data->ntp_addr);
		else if (strcmp(onvif_data->ntp_type,"IPv6") == 0)
			xmlNewTextChild(ntpManual, ns_tt, BAD_CAST "IPv6Address", BAD_CAST onvif_data->ntp_addr);
		else
			xmlNewTextChild(ntpManual, ns_tt, BAD_CAST "DNSName", BAD_CAST onvif_data->ntp_addr);
    }

    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setNTP");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setNTP - No XML reply");
    }
    return result;
}

int getHostname(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetHostname", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body//tds:GetHostnameResponse//tds:HostnameInformation//tt:FromDHCP";
        xpath = BAD_CAST "//s:Body//tds:GetHostnameResponse//tds:HostnameInformation//tt:Name";
        getXmlValue(reply, xpath, onvif_data->host_name, 128);
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getHostname");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getHostname - No XML reply");
    }
    return result;
}

int setHostname(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    if (onvif_data->host_name[0]) {
        xmlNodePtr setHostname = xmlNewTextChild(body, ns_tds, BAD_CAST "SetHostname", NULL);
        xmlNewTextChild(setHostname, ns_tds, BAD_CAST "Name", BAD_CAST onvif_data->host_name);
        /* Do I also need to set FromDHCP to false ? */
    } else {
        xmlNodePtr setHostname = xmlNewTextChild(body, ns_tds, BAD_CAST "SetHostnameFromDHCP", NULL);
        xmlNewTextChild(setHostname, ns_tds, BAD_CAST "FromDHCP", BAD_CAST "true");
    }

    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
		/* Should check for RebootNeeded=true from setHostnameFronDHCP */
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setHostname");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setHostname - No XML reply");
    }
    return result;
}


int getCapabilities(struct OnvifData *onvif_data) {
    memset(onvif_data->device_service, 0, sizeof(onvif_data->device_service));
    memset(onvif_data->event_service, 0, sizeof(onvif_data->event_service));
    memset(onvif_data->imaging_service, 0, sizeof(onvif_data->imaging_service));
    memset(onvif_data->media_service, 0, sizeof(onvif_data->media_service));
    memset(onvif_data->ptz_service, 0, sizeof(onvif_data->ptz_service));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr capabilities = xmlNewTextChild(body, ns_tds, BAD_CAST "GetCapabilities", NULL);
    xmlNewTextChild(capabilities, ns_tds, BAD_CAST "Category", BAD_CAST "All");
    char cmd[4096] = {0};


    strcpy(onvif_data->device_service, onvif_data->xaddrs);
    extractOnvifService(onvif_data->device_service, true);

    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath;

        xpath = BAD_CAST "//s:Body//tds:GetCapabilitiesResponse//tds:Capabilities//tt:Events//tt:XAddr";
        if (getXmlValue(reply, xpath, onvif_data->event_service, 1024) == 0)
            extractOnvifService(onvif_data->event_service, true);

        xpath = BAD_CAST "//s:Body//tds:GetCapabilitiesResponse//tds:Capabilities//tt:Imaging//tt:XAddr";
        if (getXmlValue(reply, xpath, onvif_data->imaging_service, 1024) == 0)
            extractOnvifService(onvif_data->imaging_service, true);

        xpath = BAD_CAST "//s:Body//tds:GetCapabilitiesResponse//tds:Capabilities//tt:Media//tt:XAddr";
        if (getXmlValue(reply, xpath, onvif_data->media_service, 1024) == 0)
            extractOnvifService(onvif_data->media_service, true);

        xpath = BAD_CAST "//s:Body//tds:GetCapabilitiesResponse//tds:Capabilities//tt:PTZ//tt:XAddr";
        if (getXmlValue(reply, xpath, onvif_data->ptz_service, 1024) == 0)
            extractOnvifService(onvif_data->ptz_service, true);

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getCapabilities");

        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getCapabilities - No XML reply");
    }
    return result;
}

int getVideoEncoderConfigurationOptions(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    for (int i = 0; i < 16; i++) {
        memset(onvif_data->resolutions_buf[i], 0, sizeof(onvif_data->resolutions_buf[i]));
    }
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getVideoEncoderConfigurationOptions = xmlNewTextChild(body, ns_trt, BAD_CAST "GetVideoEncoderConfigurationOptions", NULL);
    if (onvif_data->videoEncoderConfigurationToken[0])
        xmlNewTextChild(getVideoEncoderConfigurationOptions, ns_trt, BAD_CAST "ConfigurationToken", BAD_CAST onvif_data->videoEncoderConfigurationToken);
    if (onvif_data->profileToken[0])
        xmlNewTextChild(getVideoEncoderConfigurationOptions, ns_trt, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    char cmd[4096] = {'/0'};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *width = NULL;
        xmlChar *height = NULL;
        xmlChar *xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:H264//tt:ResolutionsAvailable";
        xmlNodeSetPtr nodeset;
        xmlXPathObjectPtr xml_result = getNodeSet(reply, xpath);
        int k = 0;
        if (xml_result) {
            nodeset = xml_result->nodesetval;
            for (int i=0; i<nodeset->nodeNr; i++) {
                xmlNodePtr cur = nodeset->nodeTab[i]->children;
                while(cur != NULL) {
                    if ((!xmlStrcmp(cur->name, (const xmlChar *) "Width"))) {
                        width = xmlNodeListGetString(reply, cur->xmlChildrenNode, 1);
                    }
                    else if ((!xmlStrcmp(cur->name, (const xmlChar *) "Height"))) {
                        height = xmlNodeListGetString(reply, cur->xmlChildrenNode, 1);
                    }
                    cur = cur->next;
                }
                char tmp[128] = {'/0'};
                if ((strlen((char *)width) + strlen((char *)height)) > 124) {
                  fprintf(stderr, "xmlNodeListString return buffer overflow %zu\n", strlen((char *)width) + strlen((char *)height));
                } else {
                  sprintf(tmp, "%s x %s", width, height);
                }

                int size = 0;
                bool found_size = false;
                while (!found_size) {
                    if (strlen(onvif_data->resolutions_buf[size]) == 0) {
                        found_size = true;
                    } else {
                        size++;
                        if (size > 15)
                            found_size = true;
                    }
                }
                bool duplicate = false;
                for (int n=0; n<size; n++) {
                    if (strcmp(onvif_data->resolutions_buf[n], tmp) == 0) {
                        duplicate = true;
                    }
                }
                if (!duplicate) {
                    strcpy(onvif_data->resolutions_buf[size], tmp);
                    k++;
                }

                if (width != NULL)
                    xmlFree(width);
                if (height != NULL)
                    xmlFree(height);
            }
            xmlXPathFreeObject(xml_result);
        }

        char temp_buf[128];
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:H264//tt:GovLengthRange//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->gov_length_min = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:H264//tt:GovLengthRange//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->gov_length_max = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:H264//tt:FrameRateRange//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->frame_rate_min = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:H264//tt:FrameRateRange//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->frame_rate_max = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:Extension//tt:H264//tt:BitrateRange//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->bitrate_min = atoi(temp_buf);
        else
            onvif_data->bitrate_min = 128;
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationOptionsResponse//trt:Options//tt:Extension//tt:H264//tt:BitrateRange//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->bitrate_max = atoi(temp_buf);
        else
            onvif_data->bitrate_max = 16384;

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getVideoEncoderConfigurationOptions");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getVideoEncoderConfigurationOptions - No XML reply");
    }
    return result;
}

int getVideoEncoderConfiguration(struct OnvifData *onvif_data) {
    memset(onvif_data->video_encoder_name, 0, sizeof(onvif_data->video_encoder_name));
    memset(onvif_data->encoding, 0, sizeof(onvif_data->encoding));
    memset(onvif_data->h264_profile, 0, sizeof(onvif_data->h264_profile));
    memset(onvif_data->multicast_address_type, 0, sizeof(onvif_data->multicast_address_type));
    memset(onvif_data->multicast_address, 0, sizeof(onvif_data->multicast_address));
    memset(onvif_data->session_time_out, 0, sizeof(onvif_data->session_time_out));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));

    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getVideoEncoderConfiguration = xmlNewTextChild(body, ns_trt, BAD_CAST "GetVideoEncoderConfiguration", NULL);
    xmlNewTextChild(getVideoEncoderConfiguration, ns_trt, BAD_CAST "ConfigurationToken", BAD_CAST onvif_data->videoEncoderConfigurationToken);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath;
        char temp_buf[128] = {0};
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Name";
        getXmlValue(reply, xpath, onvif_data->video_encoder_name, 128);

        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:UseCount";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->use_count = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:GuaranteedFrameRate";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0) {
            if (strcmp(temp_buf, "true") == 0)
                onvif_data->guaranteed_frame_rate = true;
            else
                onvif_data->guaranteed_frame_rate = false;
        }
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Encoding";
        getXmlValue(reply, xpath, onvif_data->encoding, 128);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Resolution//tt:Width";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->width = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Resolution//tt:Height";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->height = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Quality";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->quality = atof(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:RateControl//tt:FrameRateLimit";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->frame_rate = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:RateControl//tt:EncodingInterval";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->encoding_interval = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:RateControl//tt:BitrateLimit";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->bitrate = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:H264//tt:H264Profile";
        getXmlValue(reply, xpath, onvif_data->h264_profile, 128);

        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:H264//tt:GovLength";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->gov_length = atoi(temp_buf);

        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Address//tt:Type";
        getXmlValue(reply, xpath, onvif_data->multicast_address_type, 128);
		if (strcmp(onvif_data->multicast_address_type,"IPv6") == 0)
            xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Address//tt:IPv6Address";
		else
            xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Address//tt:IPv4Address";
        getXmlValue(reply, xpath, onvif_data->multicast_address, 128);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Port";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->multicast_port = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:TTL";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->multicast_ttl = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:AutoStart";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0) {
        if (strcmp(temp_buf, "true") == 0)
            onvif_data->autostart = true;
        else
            onvif_data->autostart = false;
        }
        xpath = BAD_CAST "//s:Body//trt:GetVideoEncoderConfigurationResponse//trt:Configuration//tt:SessionTimeout";
        getXmlValue(reply, xpath, onvif_data->session_time_out, 128);

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getVideoEncoderConfiguration");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getVideoEncoderConfiguration - No XML reply");
    }
    return result;
}

int setVideoEncoderConfiguration(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    char frame_rate_buf[128] = {0};
    char gov_length_buf[128] = {0};
    char bitrate_buf[128] = {0};
    char width_buf[128] = {0};
    char height_buf[128] = {0};
    char use_count_buf[128] = {0};
    char quality_buf[128] = {0};
    char multicast_port_buf[128] = {0};
    char multicast_ttl_buf[128] = {0};
    char autostart_buf[128] = {0};
    char encoding_interval_buf[128] = {0};

    sprintf(frame_rate_buf, "%d", onvif_data->frame_rate);
    sprintf(gov_length_buf, "%d", onvif_data->gov_length);
    sprintf(bitrate_buf, "%d", onvif_data->bitrate);
    sprintf(use_count_buf, "%d", onvif_data->use_count);
    sprintf(width_buf, "%d", onvif_data->width);
    sprintf(height_buf, "%d", onvif_data->height);

    sprintf(quality_buf, "%f", onvif_data->quality);
    for (int i = 0; i < strlen(quality_buf); i++) {
        if (quality_buf[i] == ',')
            quality_buf[i] = '.';
    }

    sprintf(multicast_port_buf, "%d", onvif_data->multicast_port);
    sprintf(multicast_ttl_buf, "%d", onvif_data->multicast_ttl);
    if (onvif_data->autostart)
        strcpy(autostart_buf, "true");
    else
        strcpy(autostart_buf, "false");
    sprintf(encoding_interval_buf, "%d", onvif_data->encoding_interval);

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setVideoEncoderConfiguration = xmlNewTextChild(body, ns_trt, BAD_CAST "SetVideoEncoderConfiguration", NULL);
    xmlNodePtr configuration = xmlNewTextChild(setVideoEncoderConfiguration, ns_trt, BAD_CAST "Configuration", NULL);
    xmlNewProp(configuration, BAD_CAST "token", BAD_CAST onvif_data->videoEncoderConfigurationToken);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "Name", BAD_CAST onvif_data->video_encoder_name);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "UseCount", BAD_CAST use_count_buf);
#ifdef ONVIF19060
	/* Sad, but not supported until 19.06 release - crashes my older camera */
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "GuaranteedFrameRate", onvif_data->guaranteed_frame_rate?BAD_CAST "true":BAD_CAST "false");
#endif
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "Encoding", onvif_data->encoding[0]?BAD_CAST onvif_data->encoding:BAD_CAST "H264");
    xmlNodePtr resolution = xmlNewTextChild(configuration, ns_tt, BAD_CAST "Resolution", NULL);
    xmlNewTextChild(resolution, ns_tt, BAD_CAST "Width", BAD_CAST width_buf);
    xmlNewTextChild(resolution, ns_tt, BAD_CAST "Height", BAD_CAST height_buf);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "Quality", BAD_CAST quality_buf);
    xmlNodePtr rateControl = xmlNewTextChild(configuration, ns_tt, BAD_CAST "RateControl", NULL);
    xmlNewTextChild(rateControl, ns_tt, BAD_CAST "FrameRateLimit", BAD_CAST frame_rate_buf);
    xmlNewTextChild(rateControl, ns_tt, BAD_CAST "EncodingInterval", BAD_CAST encoding_interval_buf);
    xmlNewTextChild(rateControl, ns_tt, BAD_CAST "BitrateLimit", BAD_CAST bitrate_buf);
    xmlNodePtr h264 = xmlNewTextChild(configuration, ns_tt, BAD_CAST "H264", NULL);
    xmlNewTextChild(h264, ns_tt, BAD_CAST "GovLength", BAD_CAST gov_length_buf);
    xmlNewTextChild(h264, ns_tt, BAD_CAST "H264Profile", BAD_CAST onvif_data->h264_profile);
    xmlNodePtr multicast = xmlNewTextChild(configuration, ns_tt, BAD_CAST "Multicast", NULL);
    xmlNodePtr address = xmlNewTextChild(multicast, ns_tt, BAD_CAST "Address", NULL);
    xmlNewTextChild(address, ns_tt, BAD_CAST "Type", BAD_CAST onvif_data->multicast_address_type);
    xmlNewTextChild(address, ns_tt, BAD_CAST "IPv4Address", BAD_CAST onvif_data->multicast_address);
    xmlNewTextChild(multicast, ns_tt, BAD_CAST "Port", BAD_CAST multicast_port_buf);
    xmlNewTextChild(multicast, ns_tt, BAD_CAST "TTL", BAD_CAST multicast_ttl_buf);
    xmlNewTextChild(multicast, ns_tt, BAD_CAST "AutoStart", BAD_CAST autostart_buf);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "SessionTimeout", BAD_CAST onvif_data->session_time_out);
    xmlNewTextChild(setVideoEncoderConfiguration, ns_trt, BAD_CAST "ForcePersistence", BAD_CAST "true");
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setVideoEncoderConfiguration");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setVideoEncoderConfiguration - No XML reply");
    }
    return result;
}

int getAudioEncoderConfigurationOptions(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    for (int i = 0; i < 3; i++) {
        memset(onvif_data->audio_encoders[i], 0, sizeof(onvif_data->audio_encoders[i]));
        for (int j=0; j<8; j++) {
            onvif_data->audio_sample_rates[i][j] = 0;
            onvif_data->audio_bitrates[i][j] = 0;
        }
    }
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getAudioEncoderConfigurationOptions = xmlNewTextChild(body, ns_trt, BAD_CAST "GetAudioEncoderConfigurationOptions", NULL);
    if (onvif_data->audioEncoderConfigurationToken[0])
        xmlNewTextChild(getAudioEncoderConfigurationOptions, ns_trt, BAD_CAST "ConfigurationToken", BAD_CAST onvif_data->audioEncoderConfigurationToken);
    if (onvif_data->profileToken[0])
        xmlNewTextChild(getAudioEncoderConfigurationOptions, ns_trt, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    char cmd[4096] = {'/0'};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationOptionsResponse//trt:Options//tt:Encoding";
        xmlNodeSetPtr nodeset;
        xmlXPathObjectPtr xml_result = getNodeSet(reply, xpath);
        int k = 0;
        if (xml_result) {
            nodeset = xml_result->nodesetval;
            for (int i=0; i<nodeset->nodeNr; i++) {
                xmlNodePtr cur = nodeset->nodeTab[i]->children;
                while(cur != NULL) {
                    strcpy(onvif_data->audio_encoders[i], cur->content);
                    cur = cur->next;
                }
            }
            xmlXPathFreeObject(xml_result);
        }

        xmlChar* item = NULL;

        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationOptionsResponse//trt:Options//tt:BitrateList";
        xml_result = getNodeSet(reply, xpath);
        k = 0;
        if (xml_result) {
            nodeset = xml_result->nodesetval;
            for (int i=0; i<nodeset->nodeNr; i++) {
                xmlNodePtr cur = nodeset->nodeTab[i]->children;
                int j = 0;
                while(cur != NULL) {
                    item = xmlNodeListGetString(reply, cur->xmlChildrenNode, 1);
                    if (item) {
                        onvif_data->audio_bitrates[i][j] = atoi(item);
                        j++;
                    }
                    cur = cur->next;
                }
            }
            xmlXPathFreeObject(xml_result);
        }

        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationOptionsResponse//trt:Options//tt:SampleRateList";
        xml_result = getNodeSet(reply, xpath);
        k = 0;
        if (xml_result) {
            nodeset = xml_result->nodesetval;
            for (int i=0; i<nodeset->nodeNr; i++) {
                xmlNodePtr cur = nodeset->nodeTab[i]->children;
                int j = 0;
                while(cur != NULL) {
                    item = xmlNodeListGetString(reply, cur->xmlChildrenNode, 1);
                    if (item) {
                        onvif_data->audio_sample_rates[i][j] = atoi(item);
                        j++;
                    }
                    cur = cur->next;
                }
            }
            xmlXPathFreeObject(xml_result);
        }

        if (item)
            xmlFree(item);

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getAudioEncoderConfigurationOptions");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getAudioEncoderConfigurationOptions - No XML reply");
    }
    return result;
}

int getAudioEncoderConfiguration(struct OnvifData *onvif_data) {
    memset(onvif_data->audio_name, 0, sizeof(onvif_data->audio_name));
    memset(onvif_data->audio_encoding, 0, sizeof(onvif_data->audio_encoding));
    memset(onvif_data->audio_session_timeout, 0, sizeof(onvif_data->audio_session_timeout));
    memset(onvif_data->audio_multicast_type, 0, sizeof(onvif_data->audio_multicast_type));
    memset(onvif_data->audio_multicast_address, 0, sizeof(onvif_data->audio_multicast_address));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));

    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getAudioEncoderConfiguration = xmlNewTextChild(body, ns_trt, BAD_CAST "GetAudioEncoderConfiguration", NULL);
    xmlNewTextChild(getAudioEncoderConfiguration, ns_trt, BAD_CAST "ConfigurationToken", BAD_CAST onvif_data->audioEncoderConfigurationToken);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath;
        char temp_buf[128] = {0};
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Name";
        getXmlValue(reply, xpath, onvif_data->audio_name, 128);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:UseCount";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->audio_use_count = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Encoding";
        getXmlValue(reply, xpath, onvif_data->audio_encoding, 128);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Bitrate";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->audio_bitrate = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:SampleRate";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->audio_sample_rate = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:SessionTimeout";
        getXmlValue(reply, xpath, onvif_data->audio_session_timeout, 128);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Address//tt:Type";
        getXmlValue(reply, xpath, onvif_data->audio_multicast_type, 128);
		if (strcmp(temp_buf,"IPv6") == 0)
            xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Address//tt:IPv6Address";
		else
            xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Address//tt:IPv4Address";
        getXmlValue(reply, xpath, onvif_data->audio_multicast_address, 128);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:Port";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->audio_multicast_port = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:TTL";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->audio_multicast_TTL = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//trt:GetAudioEncoderConfigurationResponse//trt:Configuration//tt:Multicast//tt:AutoStart";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0) {
            if (strcmp(temp_buf, "true") == 0) 
                onvif_data->audio_multicast_auto_start = true;
            else
                onvif_data->audio_multicast_auto_start = false;
        }

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getAudioEncoderConfiguration");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getAudioEncoderConfiguration - No XML reply");
    }
    return result;
}

int setAudioEncoderConfiguration(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    char use_count_buf[128] = {0};
    char bitrate_buf[128] = {0};
    char sample_rate_buf[123] = {0};
    char multicast_port_buf[128] = {0};
    char multicast_ttl_buf[128] = {0};
    char autostart_buf[128] = {0};

    sprintf(use_count_buf, "%d", onvif_data->audio_use_count);
    sprintf(bitrate_buf, "%d", onvif_data->audio_bitrate);
    sprintf(sample_rate_buf, "%d", onvif_data->audio_sample_rate);
    sprintf(multicast_port_buf, "%d", onvif_data->audio_multicast_port);
    sprintf(multicast_ttl_buf, "%d", onvif_data->audio_multicast_TTL);
    if (onvif_data->audio_multicast_auto_start)
        strcpy(autostart_buf, "true");
    else
        strcpy(autostart_buf, "false");

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setAudioEncoderConfiguration = xmlNewTextChild(body, ns_trt, BAD_CAST "SetAudioEncoderConfiguration", NULL);
    xmlNodePtr configuration = xmlNewTextChild(setAudioEncoderConfiguration, ns_trt, BAD_CAST "Configuration", NULL);
    xmlNewProp(configuration, BAD_CAST "token", BAD_CAST onvif_data->audioEncoderConfigurationToken);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "UseCount", BAD_CAST use_count_buf);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "Name", BAD_CAST onvif_data->audio_name);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "Encoding", BAD_CAST onvif_data->audio_encoding);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "Bitrate", BAD_CAST bitrate_buf);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "SampleRate", BAD_CAST sample_rate_buf);
    xmlNodePtr multicast = xmlNewTextChild(configuration, ns_tt, BAD_CAST "Multicast", NULL);
    xmlNodePtr address = xmlNewTextChild(multicast, ns_tt, BAD_CAST "Address", NULL);
    xmlNewTextChild(address, ns_tt, BAD_CAST "Type", BAD_CAST onvif_data->audio_multicast_type);
    xmlNewTextChild(address, ns_tt, BAD_CAST "IPv4Address", BAD_CAST onvif_data->audio_multicast_address);
    xmlNewTextChild(multicast, ns_tt, BAD_CAST "Port", BAD_CAST multicast_port_buf);
    xmlNewTextChild(multicast, ns_tt, BAD_CAST "TTL", BAD_CAST multicast_ttl_buf);
    xmlNewTextChild(multicast, ns_tt, BAD_CAST "AutoStart", BAD_CAST autostart_buf);
    xmlNewTextChild(configuration, ns_tt, BAD_CAST "SessionTimeout", BAD_CAST onvif_data->audio_session_timeout);
    xmlNewTextChild(setAudioEncoderConfiguration, ns_trt, BAD_CAST "ForcePersistence", BAD_CAST "true");
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setAudioEncoderConfiguration");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setAudioEncoderConfiguration - No XML reply");
    }
    return result;
}

int getProfile(struct OnvifData *onvif_data) {
    memset(onvif_data->videoEncoderConfigurationToken, 0, sizeof(onvif_data->videoEncoderConfigurationToken));
    memset(onvif_data->videoSourceConfigurationToken, 0, sizeof(onvif_data->videoSourceConfigurationToken));
    memset(onvif_data->audioEncoderConfigurationToken, 0, sizeof(onvif_data->audioEncoderConfigurationToken));
    memset(onvif_data->audioSourceConfigurationToken, 0, sizeof(onvif_data->audioSourceConfigurationToken));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));

    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getProfile = xmlNewTextChild(body, ns_trt, BAD_CAST "GetProfile", NULL);
    xmlNewTextChild(getProfile, ns_trt, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        char temp_buf[128];

        xmlChar *xpath;

        xpath = BAD_CAST "//s:Body//trt:GetProfileResponse//trt:Profile//tt:AudioEncoderConfiguration";
        getNodeAttribute(reply, xpath, BAD_CAST "token", onvif_data->audioEncoderConfigurationToken, 128);
        xpath = BAD_CAST "//s:Body//trt:GetProfileResponse//trt:Profile//tt:AudioSourceConfiguration//tt:SourceToken";
        getXmlValue(reply, xpath, onvif_data->audioSourceConfigurationToken, 128);

        xpath = BAD_CAST "//s:Body//trt:GetProfileResponse//trt:Profile//tt:VideoEncoderConfiguration";
        getNodeAttribute(reply, xpath, BAD_CAST "token", onvif_data->videoEncoderConfigurationToken, 128);
        xpath = BAD_CAST "//s:Body//trt:GetProfileResponse//trt:Profile//tt:VideoSourceConfiguration//tt:SourceToken";
        getXmlValue(reply, xpath, onvif_data->videoSourceConfigurationToken, 128);

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getProfile");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getProfile - No XML reply");
    }
    return result;
}

int getOptions(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_timg = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/imaging/wsdl", BAD_CAST "timg");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getOptions = xmlNewTextChild(body, ns_timg, BAD_CAST "GetOptions", NULL);
    xmlNewTextChild(getOptions, ns_timg, BAD_CAST "VideoSourceToken", BAD_CAST onvif_data->videoSourceConfigurationToken);
    char cmd[4096] = {'/0'};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->imaging_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath;
        char temp_buf[128] = {'/0'};

        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:Brightness//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->brightness_min = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:Brightness//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->brightness_max = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:ColorSaturation//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->saturation_min = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:ColorSaturation//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->saturation_max = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:Contrast//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->contrast_min = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:Contrast//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->contrast_max = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:Sharpness//tt:Min";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->sharpness_min = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetOptionsResponse//timg:ImagingOptions//tt:Sharpness//tt:Max";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->sharpness_max = atoi(temp_buf);

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getOptions");
        xmlFreeDoc(reply);
     } else {
        result = -1;
        strcpy(onvif_data->last_error, "getOptions - No XML reply");
    }
    return result;
}

int getImagingSettings(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_timg = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/imaging/wsdl", BAD_CAST "timg");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getImagingSettings = xmlNewTextChild(body, ns_timg, BAD_CAST "GetImagingSettings", NULL);
    xmlNewTextChild(getImagingSettings, ns_timg, BAD_CAST "VideoSourceToken", BAD_CAST onvif_data->videoSourceConfigurationToken);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->imaging_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlChar *xpath;
        char temp_buf[128] = {'/0'};

        xpath = BAD_CAST "//s:Body//timg:GetImagingSettingsResponse//timg:ImagingSettings//tt:Brightness";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->brightness = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetImagingSettingsResponse//timg:ImagingSettings//tt:ColorSaturation";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->saturation = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetImagingSettingsResponse//timg:ImagingSettings//tt:Contrast";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->contrast = atoi(temp_buf);
        xpath = BAD_CAST "//s:Body//timg:GetImagingSettingsResponse//timg:ImagingSettings//tt:Sharpness";
        if (getXmlValue(reply, xpath, temp_buf, 128) == 0)
            onvif_data->sharpness = atoi(temp_buf);

        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getImagingSettings");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getImagingSettings - No XML reply");
    }
    return result;
}

int setImagingSettings(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    char brightness_buf[128] = {0};
    char saturation_buf[128] = {0};
    char contrast_buf[128] = {0};
    char sharpness_buf[128] = {0};
    sprintf(brightness_buf, "%d", onvif_data->brightness);
    sprintf(saturation_buf, "%d", onvif_data->saturation);
    sprintf(contrast_buf, "%d", onvif_data->contrast);
    sprintf(sharpness_buf, "%d", onvif_data->sharpness);

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_timg = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/imaging/wsdl", BAD_CAST "timg");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setImagingSettings = xmlNewTextChild(body, ns_timg, BAD_CAST "SetImagingSettings", NULL);
    xmlNewTextChild(setImagingSettings, ns_timg, BAD_CAST "VideoSourceToken", BAD_CAST onvif_data->videoSourceConfigurationToken);
    xmlNodePtr imagingSettings = xmlNewTextChild(setImagingSettings, ns_timg, BAD_CAST "ImagingSettings", NULL);
    xmlNewTextChild(imagingSettings, ns_tt, BAD_CAST "Brightness", BAD_CAST brightness_buf);
    xmlNewTextChild(imagingSettings, ns_tt, BAD_CAST "ColorSaturation", BAD_CAST saturation_buf);
    xmlNewTextChild(imagingSettings, ns_tt, BAD_CAST "Contrast", BAD_CAST contrast_buf);
    xmlNewTextChild(imagingSettings, ns_tt, BAD_CAST "Sharpness", BAD_CAST sharpness_buf);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->imaging_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setImagingSettings");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setImagingSettings - No XML reply");
    }
    return result;
}

int continuousMove(float x, float y, float z, struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    char pan_tilt_string[128] = {0};
    char zoom_string[128] = {0};
    sprintf(pan_tilt_string, "PanTilt x=\"%.*f\" y=\"%.*f\"", 2, x, 2, y);
    sprintf(zoom_string, "Zoom x=\"%.*f\"", 2, z);

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_ptz = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/ptz/wsdl", BAD_CAST "ptz");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr continuousMove = xmlNewTextChild(body, ns_ptz, BAD_CAST "ContinuousMove", NULL);
    xmlNewTextChild(continuousMove, ns_ptz, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    xmlNodePtr velocity = xmlNewTextChild(continuousMove, ns_ptz, BAD_CAST "Velocity", BAD_CAST NULL);
    xmlNewTextChild(velocity, ns_tt, BAD_CAST pan_tilt_string, NULL);
    xmlNewTextChild(velocity, ns_tt, BAD_CAST zoom_string, NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->ptz_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " continuousMove");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "continuousMove - No XML reply");
    }
    return result;
}

int moveStop(int type, struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    char pan_tilt_flag[128] = {0};
    char zoom_flag[128] = {0};

    if (type == PAN_TILT_STOP) {
        strcpy(pan_tilt_flag, "true");
        strcpy(zoom_flag, "false");
    }
    else if (type == ZOOM_STOP) {
        strcpy(pan_tilt_flag, "false");
        strcpy(zoom_flag, "true");
    }

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_ptz = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/ptz/wsdl", BAD_CAST "ptz");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr stop = xmlNewTextChild(body, ns_ptz, BAD_CAST "Stop", NULL);
    xmlNewTextChild(stop, ns_ptz, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    xmlNewTextChild(stop, ns_ptz, BAD_CAST "PanTilt", BAD_CAST pan_tilt_flag);
    xmlNewTextChild(stop, ns_ptz, BAD_CAST "Zoom", BAD_CAST zoom_flag);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->ptz_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " moveStop");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "moveStop - No XML reply");
    }
    return result;
}

int setPreset(char *arg, struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_ptz = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/ptz/wsdl", BAD_CAST "ptz");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setPreset = xmlNewTextChild(body, ns_ptz, BAD_CAST "SetPreset", NULL);
    xmlNewTextChild(setPreset, ns_ptz, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    xmlNewTextChild(setPreset, ns_ptz, BAD_CAST "PresetToken", BAD_CAST arg);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->ptz_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setPreset");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setPreset - No XML reply");
    }
    return result;
}

int gotoPreset(char *arg, struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_ptz = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver20/ptz/wsdl", BAD_CAST "ptz");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr gotoPreset = xmlNewTextChild(body, ns_ptz, BAD_CAST "GotoPreset", NULL);
    xmlNewTextChild(gotoPreset, ns_ptz, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    xmlNewTextChild(gotoPreset, ns_ptz, BAD_CAST "PresetToken", BAD_CAST arg);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->ptz_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " gotoPreset");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "gotoPreset - No XML reply");
    }
    return result;
}

int setUser(char *new_password, struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setUser = xmlNewTextChild(body, ns_tds, BAD_CAST "SetUser", NULL);
    xmlNodePtr user = xmlNewTextChild(setUser, ns_tds, BAD_CAST "User", NULL);
    xmlNewTextChild(user, ns_tt, BAD_CAST "Username", BAD_CAST "admin");
    xmlNewTextChild(user, ns_tt, BAD_CAST "Password", BAD_CAST new_password);
    xmlNewTextChild(user, ns_tt, BAD_CAST "UserLevel", BAD_CAST "Administrator");
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setUser");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setUser - No XML reply");
    }
    return result;
}

int setSystemDateAndTime(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    time_t rawtime;
    time(&rawtime);
    struct tm *UTCTime = localtime(&rawtime);
    char dst_flag_buf[128];
    if (UTCTime->tm_isdst == 1)
        strcpy(dst_flag_buf, "true");
    else
        strcpy(dst_flag_buf, "false");
    char hour_buf[128];
    char minute_buf[128];
    char second_buf[128];
    char year_buf[128];
    char month_buf[128];
    char day_buf[128];
    sprintf(hour_buf, "%d", UTCTime->tm_hour);
    sprintf(minute_buf, "%d", UTCTime->tm_min);
    sprintf(second_buf, "%d", UTCTime->tm_sec);
    sprintf(year_buf, "%d", UTCTime->tm_year + 1900);
    sprintf(month_buf, "%d", UTCTime->tm_mon + 1);
    sprintf(day_buf, "%d", UTCTime->tm_mday);

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setSystemDateAndTime = xmlNewTextChild(body, ns_tds, BAD_CAST "SetSystemDateAndTime", NULL);
    xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "DateTimeType", BAD_CAST "Manual");
    xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "DaylightSavings", BAD_CAST dst_flag_buf);
    xmlNodePtr timeZone = xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "TimeZone", NULL);
    xmlNewTextChild(timeZone, ns_tt, BAD_CAST "TZ", BAD_CAST "UTC0");
    xmlNodePtr utcDateTime = xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "UTCDateTime", NULL);
    xmlNodePtr cameraTime = xmlNewTextChild(utcDateTime, ns_tt, BAD_CAST "Time", NULL);
    xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Hour", BAD_CAST hour_buf);
    xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Minute", BAD_CAST minute_buf);
    xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Second", BAD_CAST second_buf);
    xmlNodePtr cameraDate = xmlNewTextChild(utcDateTime, ns_tt, BAD_CAST "Date", NULL);
    xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Year", BAD_CAST year_buf);
    xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Month", BAD_CAST month_buf);
    xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Day", BAD_CAST day_buf);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " setSystemDateAndTime");
        xmlFreeDoc(reply);
    }
    else {
        result = -1;
        strcpy(onvif_data->last_error, "setSystemDateAndTime - No XML reply");
    }
    return result;
}

int setSystemDateAndTimeUsingTimezone(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    time_t rawtime;
    time(&rawtime);
	bool special = false;
    struct tm *UTCTime = localtime(&rawtime);
    char dst_flag_buf[128];
    if (UTCTime->tm_isdst == 1)
        strcpy(dst_flag_buf, "true");
    else
        strcpy(dst_flag_buf, "false");
    if (strcmp(onvif_data->timezone,"UTC0") == 0) {
        special = true;
    } else {
        if (!onvif_data->timezone[0]) {
#ifndef _WIN32
            // work out a timezone to use on the camera 
            int h = -(UTCTime->tm_gmtoff/3600);
            int m = (UTCTime->tm_gmtoff + 3600 * h)/60;
            if (m)
                sprintf(onvif_data->timezone,"%s%d:%02d:00%s",tzname[0],h,m,tzname[1]);
            else
                sprintf(onvif_data->timezone,"%s%d%s",tzname[0],h,tzname[1]);
#else
            int h = _timezone/3600;
            int m = (_timezone - 3600 * h)/60;
            if (m)
                sprintf(onvif_data->timezone,"%s%d:%02d:00%s",_tzname[0],h,m,_tzname[1]);
            else
                sprintf(onvif_data->timezone,"%s%d%s",_tzname[0],h,_tzname[1]);
#endif
        }
        UTCTime = gmtime(&rawtime);
    }
    if (!onvif_data->datetimetype)
        onvif_data->datetimetype = 'M'; // manual 
    char hour_buf[128];
    char minute_buf[128];
    char second_buf[128];
    char year_buf[128];
    char month_buf[128];
    char day_buf[128];
    sprintf(hour_buf, "%d", UTCTime->tm_hour);
    sprintf(minute_buf, "%d", UTCTime->tm_min);
    sprintf(second_buf, "%d", UTCTime->tm_sec);
    sprintf(year_buf, "%d", UTCTime->tm_year + 1900);
    sprintf(month_buf, "%d", UTCTime->tm_mon + 1);
    sprintf(day_buf, "%d", UTCTime->tm_mday);

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setSystemDateAndTime = xmlNewTextChild(body, ns_tds, BAD_CAST "SetSystemDateAndTime", NULL);
    xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "DateTimeType", BAD_CAST "Manual");
    xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "DaylightSavings", BAD_CAST dst_flag_buf);
    xmlNodePtr timeZone = xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "TimeZone", NULL);
    xmlNewTextChild(timeZone, ns_tt, BAD_CAST "TZ", BAD_CAST onvif_data->timezone);
    xmlNodePtr utcDateTime = xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "UTCDateTime", NULL);
    xmlNodePtr cameraTime = xmlNewTextChild(utcDateTime, ns_tt, BAD_CAST "Time", NULL);
    xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Hour", BAD_CAST hour_buf);
    xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Minute", BAD_CAST minute_buf);
    xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Second", BAD_CAST second_buf);
    xmlNodePtr cameraDate = xmlNewTextChild(utcDateTime, ns_tt, BAD_CAST "Date", NULL);
    xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Year", BAD_CAST year_buf);
    xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Month", BAD_CAST month_buf);
    xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Day", BAD_CAST day_buf);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        xmlFreeDoc(reply);
        if (result == 0 && onvif_data->datetimetype == 'N') {
            // switch back to NTP after we have nudged it to correct 
            time_t newtime;
            time(&newtime);
            if (newtime != rawtime) {
                // save a little effort if we are within a second of the previous check 
                if (special)
                    UTCTime = localtime(&newtime);
                else
                    UTCTime = gmtime(&newtime);
                sprintf(hour_buf, "%d", UTCTime->tm_hour);
                sprintf(minute_buf, "%d", UTCTime->tm_min);
                sprintf(second_buf, "%d", UTCTime->tm_sec);
                sprintf(year_buf, "%d", UTCTime->tm_year + 1900);
                sprintf(month_buf, "%d", UTCTime->tm_mon + 1);
                sprintf(day_buf, "%d", UTCTime->tm_mday);
            }
            doc = xmlNewDoc(BAD_CAST "1.0");
            xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
            xmlDocSetRootElement(doc, root);
            xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
            xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
            xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
            xmlSetNs(root, ns_env);
            addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
            xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
            xmlNodePtr setSystemDateAndTime = xmlNewTextChild(body, ns_tds, BAD_CAST "SetSystemDateAndTime", NULL);
            xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "DateTimeType", BAD_CAST "NTP");
            xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "DaylightSavings", BAD_CAST dst_flag_buf);
            xmlNodePtr timeZone = xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "TimeZone", NULL);
            xmlNewTextChild(timeZone, ns_tt, BAD_CAST "TZ", BAD_CAST onvif_data->timezone);
            // Need to include date/time even though the specs say it should be ignored 
            xmlNodePtr utcDateTime = xmlNewTextChild(setSystemDateAndTime, ns_tds, BAD_CAST "UTCDateTime", NULL);
            xmlNodePtr cameraTime = xmlNewTextChild(utcDateTime, ns_tt, BAD_CAST "Time", NULL);
            xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Hour", BAD_CAST hour_buf);
            xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Minute", BAD_CAST minute_buf);
            xmlNewTextChild(cameraTime, ns_tt, BAD_CAST "Second", BAD_CAST second_buf);
            xmlNodePtr cameraDate = xmlNewTextChild(utcDateTime, ns_tt, BAD_CAST "Date", NULL);
            xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Year", BAD_CAST year_buf);
            xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Month", BAD_CAST month_buf);
            xmlNewTextChild(cameraDate, ns_tt, BAD_CAST "Day", BAD_CAST day_buf);
            char cmd[4096] = {0};
            addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
            xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
            if (reply != NULL) {
                result = checkForXmlErrorMsg(reply, onvif_data->last_error);
		        xmlFreeDoc(reply);
            } else {
                result = -1;
		        strcpy(onvif_data->last_error, "setSystemDateAndTimeUsingTimezone - No XML reply");
            }
        }
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "setSystemDateAndTimeUsingTimezone 2 - No XML reply");
    }
    return result;
}

int getProfileToken(struct OnvifData *onvif_data, int profileIndex) {
    int result = 0;
    //onvif_data->profileToken[0] = 0;
    memset(onvif_data->profileToken, 0, sizeof(onvif_data->profileToken));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_trt, BAD_CAST "GetProfiles", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        getNodeAttributen(reply, BAD_CAST "//s:Body//trt:GetProfilesResponse//trt:Profiles", BAD_CAST "token", onvif_data->profileToken, 128, profileIndex);
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getProfileToken");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getProfileToken - No XML reply");
    }
    return result;
}

int getTimeOffset(struct OnvifData *onvif_data) {
    memset(onvif_data->timezone, 0, sizeof(onvif_data->timezone));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;

    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetSystemDateAndTime", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);

    if (reply != NULL) {
        char hour_buf[16] = {'/0'};
        char min_buf[16] = {'/0'};
        char sec_buf[16] = {'/0'};
        char year_buf[16] = {'/0'};
        char month_buf[16] = {'/0'};
        char day_buf[16] = {'/0'};
        char dst_buf[16] = {'/0'};
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:UTCDateTime//tt:Time//tt:Hour", hour_buf, 16);
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:UTCDateTime//tt:Time//tt:Minute", min_buf, 16);
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:UTCDateTime//tt:Time//tt:Second", sec_buf, 16);
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:UTCDateTime//tt:Date//tt:Year", year_buf, 16);
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:UTCDateTime//tt:Date//tt:Month", month_buf, 16);
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:UTCDateTime//tt:Date//tt:Day", day_buf, 16);
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:DaylightSavings", dst_buf, 16);

    	onvif_data->dst = false;
        int is_dst = 0;
        if (strcmp(dst_buf, "true") == 0) {
            is_dst = 1;
	        onvif_data->dst = true;
	    }

        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:TimeZone//tt:TZ", onvif_data->timezone, 128);
	    char dttype[16] = {'/0'};
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetSystemDateAndTimeResponse//tds:SystemDateAndTime//tt:DateTimeType", dttype, 16);
	    onvif_data->datetimetype = dttype[0]; /* M == Manual, N == NTP */

        time_t now = time(NULL);
        time_t utc_time_here = now;
	    bool special = false;
	    if (strcmp(onvif_data->timezone,"UTC0") == 0) {
	        /* special case - camera is running on local time believing it is UTC */
	        special = true;
            struct tm *utc_here = gmtime(&now);
            utc_here->tm_isdst = -1;
            utc_time_here = mktime(utc_here);
	    }

        struct tm *utc_there = localtime(&now);
        utc_there->tm_year = atoi(year_buf) - 1900;
        utc_there->tm_mon = atoi(month_buf) - 1;
        utc_there->tm_mday = atoi(day_buf);
        utc_there->tm_hour = atoi(hour_buf);
        utc_there->tm_min = atoi(min_buf);
        utc_there->tm_sec = atoi(sec_buf);
        utc_there->tm_isdst = is_dst;
	    time_t utc_time_there;
	    if (special)
	        utc_time_there = mktime(utc_there);
	    else
#ifndef _WIN32
	        utc_time_there = timegm(utc_there);
#else
	        utc_time_there = _mkgmtime(utc_there);
#endif
	    onvif_data->time_offset = utc_time_there - utc_time_here;
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getTimeOffset");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getTimeOffset - No XML reply");
    }

    return result;
}

int getStreamUri(struct OnvifData *onvif_data) {
    memset(onvif_data->stream_uri, 0, sizeof(onvif_data->stream_uri));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlNsPtr ns_tt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/schema", BAD_CAST "tt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr getStreamUri = xmlNewTextChild(body, ns_trt, BAD_CAST "GetStreamUri", NULL);
    xmlNodePtr streamSetup = xmlNewTextChild(getStreamUri, ns_trt, BAD_CAST "StreamSetup", NULL);
    xmlNewTextChild(streamSetup, ns_tt, BAD_CAST "Stream", BAD_CAST "RTP-Unicast");
    xmlNodePtr transport = xmlNewTextChild(streamSetup, ns_tt, BAD_CAST "Transport", NULL);
    xmlNewTextChild(transport, ns_tt, BAD_CAST "Protocol", BAD_CAST "RTSP");
    xmlNewTextChild(getStreamUri, ns_trt, BAD_CAST "ProfileToken", BAD_CAST onvif_data->profileToken);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        getXmlValue(reply, BAD_CAST "//s:Body//trt:GetStreamUriResponse//trt:MediaUri//tt:Uri", onvif_data->stream_uri, 1024);
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getStreamUri");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getStreamUri - No XML reply");
    }
    return result;
}

int getDeviceInformation(struct OnvifData *onvif_data) {
    memset(onvif_data->serial_number, 0, sizeof(onvif_data->serial_number));
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetDeviceInformation", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        getXmlValue(reply, BAD_CAST "//s:Body//tds:GetDeviceInformationResponse//tds:SerialNumber", onvif_data->serial_number, 128);
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " getdeviceInformation");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "getDeviceInformation - No XML reply");
    }
    return result;
}

void getDiscoveryXml2(char buffer[], int buf_size) {
    char *xml_string = "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\" xmlns:a=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\"><s:Header><a:Action s:mustUnderstand=\"1\">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action><a:MessageID>uuid:6bbdae2d-f229-42c8-a27b-93880fb80826</a:MessageID><a:ReplyTo><a:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:Address></a:ReplyTo><a:To s:mustUnderstand=\"1\">urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To></s:Header><s:Body><Probe xmlns=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\"><d:Types xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\" xmlns:dp0=\"http://www.onvif.org/ver10/device/wsdl\">dp0:Device</d:Types></Probe></s:Body></s:Envelope>";
    strcpy(buffer, xml_string);
}

void getDiscoveryXml(char buffer[], int buf_size, char uuid[47]) {
    for (int i=0; i<buf_size; i++)
        buffer[i] = '\0';
    getUUID(uuid);
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNewProp(root, BAD_CAST "xmlns:SOAP-ENV", BAD_CAST "http://www.w3.org/2003/05/soap-envelope");
    xmlNewProp(root, BAD_CAST "xmlns:a", BAD_CAST "http://schemas.xmlsoap.org/ws/2004/08/addressing");
    xmlNsPtr ns_env = xmlNewNs(root, NULL, BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_a = xmlNewNs(root, NULL, BAD_CAST "a");
    xmlSetNs(root, ns_env);
    xmlNodePtr header = xmlNewTextChild(root, ns_env, BAD_CAST "Header", NULL);
    xmlNodePtr action = xmlNewTextChild(header, ns_a, BAD_CAST "Action", BAD_CAST "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe");
    xmlNewProp(action, BAD_CAST "SOAP-ENV:mustUnderstand", BAD_CAST "1");
    xmlNodePtr messageid = xmlNewTextChild(header, ns_a, BAD_CAST "MessageID", BAD_CAST uuid);
    xmlNodePtr replyto = xmlNewTextChild(header, ns_a, BAD_CAST "ReplyTo", NULL);
    xmlNodePtr address = xmlNewTextChild(replyto, ns_a, BAD_CAST "Address", BAD_CAST "http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous");
    xmlNodePtr to = xmlNewTextChild(header, ns_a, BAD_CAST "To", BAD_CAST "urn:schemas-xmlsoap-org:ws:2005:04:discovery");
    xmlNewProp(to, BAD_CAST "SOAP-ENV:mustUnderstand", BAD_CAST "1");
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr probe = xmlNewTextChild(body, NULL, BAD_CAST "Probe", NULL);
    xmlNewProp(probe, BAD_CAST "xmlns:p", BAD_CAST "http://schemas.xmlsoap.org/ws/2005/04/discovery");
    xmlNsPtr ns_p = xmlNewNs(probe, NULL, BAD_CAST "p");
    xmlSetNs(probe, ns_p);
    xmlNodePtr types = xmlNewTextChild(probe, NULL, BAD_CAST "Types", BAD_CAST "dp0:NetworkVideoTransmitter");
    xmlNewProp(types, BAD_CAST "xmlns:d", BAD_CAST "http://schemas.xmlsoap.org/ws/2005/04/discovery");
    xmlNewProp(types, BAD_CAST "xmlns:dp0", BAD_CAST "http://www.onvif.org/ver10/network/wsdl");
    xmlNsPtr ns_d = xmlNewNs(types, NULL, BAD_CAST "d");
    xmlSetNs(types, ns_d);
    xmlOutputBufferPtr outputbuffer = xmlAllocOutputBuffer(NULL);
    xmlNodeDumpOutput(outputbuffer, doc, root, 0, 0, NULL);
    int size = xmlOutputBufferGetSize(outputbuffer);
    strcpy(buffer, (char*)xmlOutputBufferGetContent(outputbuffer));
    xmlOutputBufferFlush(outputbuffer);
    xmlOutputBufferClose(outputbuffer);
    xmlFreeDoc(doc);
}

int rebootCamera(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "SystemReboot", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " rebootCamera");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "rebootCamera - No XML reply");
    }
  return result;
}

int hardReset(struct OnvifData *onvif_data) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNodePtr setSystemFactoryDefault = xmlNewTextChild(body, ns_tds, BAD_CAST "SetSystemFactoryDefault", NULL);
    xmlNewTextChild(setSystemFactoryDefault, ns_tds, BAD_CAST "FactoryDefault", BAD_CAST "Hard");
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        result = checkForXmlErrorMsg(reply, onvif_data->last_error);
        if (result < 0)
            strcat(onvif_data->last_error, " hardReset");
        xmlFreeDoc(reply);
    } else {
        result = -1;
        strcpy(onvif_data->last_error, "hardReset - No XML reply");
    }
    return result;
}

void saveSystemDateAndTime(char *filename, struct OnvifData *onvif_data) {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetSystemDateAndTime", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlSaveFormatFile(filename, reply, 1);
        xmlFreeDoc(reply);
    }
}

void saveScopes(char *filename, struct OnvifData *onvif_data) {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetScopes", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlSaveFormatFile(filename, reply, 1);
        xmlFreeDoc(reply);
    }
}

void saveDeviceInformation(char *filename, struct OnvifData *onvif_data) {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetDeviceInformation", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlSaveFormatFile(filename, reply, 1);
        xmlFreeDoc(reply);
    }
}

void saveCapabilities(char *filename, struct OnvifData *onvif_data) {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_tds = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/device/wsdl", BAD_CAST "tds");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_tds, BAD_CAST "GetCapabilities", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->device_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlSaveFormatFile(filename, reply, 1);
        xmlFreeDoc(reply);
    }
}

void saveProfiles(char *filename, struct OnvifData *onvif_data) {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_trt, BAD_CAST "GetProfiles", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlSaveFormatFile(filename, reply, 1);
        xmlFreeDoc(reply);
    }
}

void saveServiceCapabilities(char *filename, struct OnvifData *onvif_data) {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root = xmlNewDocNode(doc, NULL, BAD_CAST "Envelope", NULL);
    xmlDocSetRootElement(doc, root);
    xmlNsPtr ns_env = xmlNewNs(root, BAD_CAST "http://www.w3.org/2003/05/soap-envelope", BAD_CAST "SOAP-ENV");
    xmlNsPtr ns_trt = xmlNewNs(root, BAD_CAST "http://www.onvif.org/ver10/media/wsdl", BAD_CAST "trt");
    xmlSetNs(root, ns_env);
    addUsernameDigestHeader(root, ns_env, onvif_data->username, onvif_data->password, onvif_data->time_offset);
    xmlNodePtr body = xmlNewTextChild(root, ns_env, BAD_CAST "Body", NULL);
    xmlNewTextChild(body, ns_trt, BAD_CAST "GetServiceCapabilities", NULL);
    char cmd[4096] = {0};
    addHttpHeader(doc, root, onvif_data->xaddrs, onvif_data->media_service, cmd, 4096);
    xmlDocPtr reply = sendCommandToCamera(cmd, onvif_data->xaddrs);
    if (reply != NULL) {
        xmlSaveFormatFile(filename, reply, 1);
        xmlFreeDoc(reply);
    }
}

int getXmlValue(xmlDocPtr doc, xmlChar *xpath, char buf[], int buf_length) {
    xmlXPathContextPtr context = xmlXPathNewContext(doc);

    if (!context) return -1;

    xmlXPathRegisterNs(context, BAD_CAST "s", BAD_CAST "http://www.w3.org/2003/05/soap-envelope");
    xmlXPathRegisterNs(context, BAD_CAST "trt", BAD_CAST "http://www.onvif.org/ver10/media/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "tt", BAD_CAST "http://www.onvif.org/ver10/schema");
    xmlXPathRegisterNs(context, BAD_CAST "tds", BAD_CAST "http://www.onvif.org/ver10/device/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "timg", BAD_CAST "http://www.onvif.org/ver20/imaging/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "wsa5", BAD_CAST "http://www.w3.org/2005/08/addressing");
    xmlXPathRegisterNs(context, BAD_CAST "wsnt", BAD_CAST "http://docs.oasis-open.org/wsn/b-2");
    xmlXPathRegisterNs(context, BAD_CAST "d", BAD_CAST "http://schemas.xmlsoap.org/ws/2005/04/discovery");
    xmlXPathRegisterNs(context, BAD_CAST "ter", BAD_CAST "http://www.onvif.org/ver10/error");
    xmlXPathRegisterNs(context, BAD_CAST "a", BAD_CAST "http://schemas.xmlsoap.org/ws/2004/08/addressing");

    xmlXPathObjectPtr result = xmlXPathEvalExpression(xpath, context);
    xmlXPathFreeContext(context);

    if (!result) return -2;

    if (xmlXPathNodeSetIsEmpty(result->nodesetval)) {
        if ((strcmp((char*) xpath, "//s:Body//s:Fault//s:Code//s:Subcode//s:Value") != 0) && (strcmp((char*) xpath, "//s:Body//s:Fault//s:Reason//s:Text") != 0)) { }
        xmlXPathFreeObject(result);
        return -3;
    }

    xmlChar* keyword = xmlNodeListGetString(doc, result->nodesetval->nodeTab[0]->xmlChildrenNode, 1);
    if (keyword) {
        memset(buf, 0, buf_length);
        strncpy(buf, (char*) keyword, buf_length);
        xmlFree(keyword);
    }

    xmlXPathFreeObject(result);
    return 0;
}

int getNodeAttributen (xmlDocPtr doc, xmlChar *xpath, xmlChar *attribute, char buf[], int buf_length, int profileIndex) {
    xmlChar *keyword = NULL;
    xmlXPathContextPtr context = xmlXPathNewContext(doc);
    if (context == NULL) {
        return -1;
    }
    xmlXPathRegisterNs(context, BAD_CAST "s", BAD_CAST "http://www.w3.org/2003/05/soap-envelope");
    xmlXPathRegisterNs(context, BAD_CAST "trt", BAD_CAST "http://www.onvif.org/ver10/media/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "tt", BAD_CAST "http://www.onvif.org/ver10/schema");
    xmlXPathRegisterNs(context, BAD_CAST "tds", BAD_CAST "http://www.onvif.org/ver10/device/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "timg", BAD_CAST "http://www.onvif.org/ver20/imaging/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "wsa5", BAD_CAST "http://www.w3.org/2005/08/addressing");
    xmlXPathRegisterNs(context, BAD_CAST "wsnt", BAD_CAST "http://docs.oasis-open.org/wsn/b-2");
    xmlXPathRegisterNs(context, BAD_CAST "ter", BAD_CAST "http://www.onvif.org/ver10/error");
    xmlXPathRegisterNs(context, BAD_CAST "a", BAD_CAST "http://schemas.xmlsoap.org/ws/2004/08/addressing");

    xmlXPathObjectPtr result = xmlXPathEvalExpression(xpath, context);
    xmlXPathFreeContext(context);
    if (result == NULL) {
        return -2;
    }

    if (xmlXPathNodeSetIsEmpty(result->nodesetval)) {
        if (result) xmlXPathFreeObject(result);
        return -3;
    }

    if (result) {
        if( profileIndex >= result->nodesetval->nodeNr )
            return -5;

        keyword = xmlGetProp(result->nodesetval->nodeTab[profileIndex], attribute);
        if (keyword != NULL) {
            if (strlen((char*) keyword) > buf_length-1) {
                xmlXPathFreeObject(result);
                xmlFree(keyword);
                return -4;
            } else {
                for (int i=0; i<buf_length; i++)
                    buf[i] = '\0';
                strcpy(buf, (char*) keyword);
            }
        }
    }

    xmlXPathFreeObject(result);
    if (keyword != NULL)
        xmlFree(keyword);
    return 0;
}

xmlXPathObjectPtr getNodeSet (xmlDocPtr doc, xmlChar *xpath) {
    xmlXPathContextPtr context;
    xmlXPathObjectPtr result;

    context = xmlXPathNewContext(doc);
    if (context == NULL) {
        return NULL;
    }
    xmlXPathRegisterNs(context, BAD_CAST "s", BAD_CAST "http://www.w3.org/2003/05/soap-envelope");
    xmlXPathRegisterNs(context, BAD_CAST "trt", BAD_CAST "http://www.onvif.org/ver10/media/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "tt", BAD_CAST "http://www.onvif.org/ver10/schema");
    xmlXPathRegisterNs(context, BAD_CAST "tds", BAD_CAST "http://www.onvif.org/ver10/device/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "timg", BAD_CAST "http://www.onvif.org/ver20/imaging/wsdl");
    xmlXPathRegisterNs(context, BAD_CAST "wsa5", BAD_CAST "http://www.w3.org/2005/08/addressing");
    xmlXPathRegisterNs(context, BAD_CAST "wsnt", BAD_CAST "http://docs.oasis-open.org/wsn/b-2");

    result = xmlXPathEvalExpression(xpath, context);
    xmlXPathFreeContext(context);
    if (result == NULL) {
        return NULL;
    }
    if(xmlXPathNodeSetIsEmpty(result->nodesetval)){
        if (result) xmlXPathFreeObject(result);
        return NULL;
    }
    return result;
}

xmlDocPtr sendCommandToCamera(char *cmd, char *xaddrs) {
    int sock = 0, valread, flags;
    const int buffer_size = 4096;
    struct sockaddr_in serv_addr;
    char buffer[4096] = {0};

    char tmp[128] = {0};
    char *mark = strstr(xaddrs, "//");
    int start = mark-xaddrs+2;
    int tmp_len = strlen(xaddrs);
    int j;
    for (j=0; j<tmp_len-start; j++) {
        if (j < 128)
            tmp[j] = xaddrs[j+start];
    }
    tmp[j] = '\0';

    mark = strstr(tmp, "/");
    int end = mark-tmp;
    char tmp2[128] = {0};
    for (j=0; j<end; j++) {
        tmp2[j] = tmp[j];
    }
    tmp2[j] = '\0';

    char host[128] = {0};
    char port_buf[128] = {0};
    mark = strstr(tmp2, ":");
    if (mark == NULL) {
        strcpy(host, tmp2);
        strcpy(port_buf, "80");
    } else {
        start = mark-tmp2;
        for (j=0; j<start; j++) {
            host[j] = tmp2[j];
        }
        host[j] = '\0';
        tmp_len = strlen(tmp2);
        for (j=start+1; j<tmp_len; j++) {
            port_buf[j-(start+1)] = tmp2[j];
        }
        port_buf[j-(start+1)] = '\0';
    }

    int port = atoi(port_buf);

#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        return NULL;
    }

    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    serv_addr.sin_addr.s_addr = inet_addr(host);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        xmlDocPtr doc = NULL;
        xmlNodePtr root_node = NULL;
        doc = xmlNewDoc(BAD_CAST "1.0");
        root_node = xmlNewNode(NULL, BAD_CAST "root");
        xmlDocSetRootElement(doc, root_node);
        xmlNewChild(root_node, NULL, BAD_CAST "error", BAD_CAST "Network error, unable to connect");
        return doc;
    }

    if (send(sock , cmd , strlen(cmd) , 0 ) < 0) {
        printf("SEND ERROR %s\n", xaddrs);
        return NULL;
    }

    char http_terminate[5];
    http_terminate[0] = '\r';
    http_terminate[1] = '\n';
    http_terminate[2] = '\r';
    http_terminate[3] = '\n';
    http_terminate[4] = '\0';

    int loop = 10;
    valread = 0;

    while (loop-- > 0) {
        int nvalread = recv( sock , buffer + valread, 4096 - 1 - valread, 0);
        if (nvalread <= 0) {
            break;
        }
        valread += nvalread;

        char * substr = strstr(buffer, http_terminate);
        if (substr) {
            break;
        }
    }

    char * substr = strstr(buffer, http_terminate);
    if (substr == NULL)
        return NULL;

    int i;
    int xml_start = substr - buffer + 4;
    if (xml_start > 1024)
        return NULL;
    char http_header[1024];
    for (i=0; i<xml_start; i++) {
        http_header[i] = buffer[i];
    }
    http_header[xml_start] = '\0';

    substr = strstr(http_header, "Content-Length: ");
    if (substr == NULL)
        return NULL;

    int length_start = substr - http_header + 16;
    if ((xml_start - length_start) > 1024)
        return NULL;
    char str_xml_length[1024];
    for (i=length_start; i<xml_start; i++) {
        if (http_header[i] == '\r' && http_header[i+1] == '\n') {
            str_xml_length[i - length_start] = '\0';
            break;
        } else {
            str_xml_length[i - length_start] = http_header[i];
        }
    }
    int xml_length = (int) strtol(str_xml_length, (char **)NULL, 10);
    if (xml_length > 65536)
        return NULL;
    char xml_reply[65536];

    for (i=0; i<valread-xml_start; i++) {
        xml_reply[i] = buffer[i+xml_start];
    }

    int cumulative_read = valread - xml_start;
    while (cumulative_read < xml_length) {
        valread = recv(sock, buffer, buffer_size, 0);
        for (i=0; i<valread; i++) {
            xml_reply[i+cumulative_read] = buffer[i];
        }
        cumulative_read = cumulative_read + valread;
    }
    xml_reply[xml_length] = '\0';

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif

    xmlDocPtr reply = xmlParseMemory(xml_reply, xml_length);
    char error_msg[1024] = {0};

    if (dump_reply) {
        dumpReply(reply);
    }

    return reply;
}

int checkForXmlErrorMsg(xmlDocPtr doc, char error_msg[1024]) {
    if (getXmlValue(doc, BAD_CAST "//s:Body//s:Fault//s:Code//s:Subcode//s:Value", error_msg, 1024) == 0) {
        return -1;
    }
    else if (getXmlValue(doc, BAD_CAST "//s:Body//s:Fault//s:Reason//s:Text", error_msg, 1024) == 0) {
        return -1;
    }
    else {
        xmlNode* root = xmlDocGetRootElement(doc);
        if (root) {
            xmlNodePtr msg = root->xmlChildrenNode;
            if ((!xmlStrcmp(msg->name, (const xmlChar *)"error"))) {
                memset(error_msg, 0, sizeof(error_msg));
                strcpy(error_msg, (char*) xmlNodeGetContent(msg));
                return -1;
            }
        }
    }
    return 0;
}

void addUsernameDigestHeader(xmlNodePtr root, xmlNsPtr ns_env, char *user, char *password, time_t offset) {
    srand (time(NULL));

#ifdef _WIN32
    _setmode(0, O_BINARY);
#endif

    unsigned int nonce_chunk_size = 20;
    unsigned char nonce_buffer[20];
    char nonce_base64[1024] = {0};
    char time_holder[1024] = {0};
    char digest_base64[1024] = {0};

    for (int i=0; i<nonce_chunk_size; i++) {
        nonce_buffer[i] = (unsigned char)rand();
    }

    unsigned char nonce_result[30];

    getBase64(nonce_buffer, nonce_chunk_size, nonce_result);
    strcpy(nonce_base64, (const char *)nonce_result);

    char time_buffer[1024];
    time_t now = time(NULL);
    now = now + offset;
    size_t time_buffer_length = strftime(time_buffer, 1024, "%Y-%m-%dT%H:%M:%S.", gmtime(&now));
    time_buffer[time_buffer_length] = '\0';
    int millisec;
    struct timeval tv;
#ifdef _WIN32
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tv.tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tv.tv_usec = (long) (system_time.wMilliseconds * 1000);
#else
    gettimeofday(&tv, NULL);
#endif
    millisec = tv.tv_usec/1000.0;
    char milli_buf[16] = {0};
    sprintf(milli_buf, "%03dZ", millisec);
    strcat(time_buffer, milli_buf);

    unsigned char hash[20];

    SHA1_CTX ctx;
    SHA1Init(&ctx);
    SHA1Update(&ctx, nonce_buffer, nonce_chunk_size);
    SHA1Update(&ctx, (const unsigned char *)time_buffer, strlen(time_buffer));
    SHA1Update(&ctx, (const unsigned char *)password, strlen(password));
    SHA1Final(hash, &ctx);

    unsigned int digest_chunk_size = SHA1_DIGEST_SIZE;
    unsigned char digest_result[128];
    getBase64(hash, digest_chunk_size, digest_result);

    strcpy(time_holder, time_buffer);
    strcpy(digest_base64, (const char *)digest_result);

    xmlNsPtr ns_wsse = xmlNewNs(root, BAD_CAST "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd", BAD_CAST "wsse");
    xmlNsPtr ns_wsu = xmlNewNs(root, BAD_CAST "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd", BAD_CAST "wsu");
    xmlNodePtr header = xmlNewTextChild(root, ns_env, BAD_CAST "Header", NULL);
    xmlNodePtr security = xmlNewTextChild(header, ns_wsse, BAD_CAST "Security", NULL);
    xmlNewProp(security, BAD_CAST "SOAP-ENV:mustUnderstand", BAD_CAST "1");
    xmlNodePtr username_token = xmlNewTextChild(security, ns_wsse, BAD_CAST "UsernameToken", NULL);
    xmlNewTextChild(username_token, ns_wsse, BAD_CAST "Username", BAD_CAST user);
    xmlNodePtr pwd = xmlNewTextChild(username_token, ns_wsse, BAD_CAST "Password", BAD_CAST digest_base64);
    xmlNewProp(pwd, BAD_CAST "Type", BAD_CAST "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest");
    xmlNodePtr nonce = xmlNewTextChild(username_token, ns_wsse, BAD_CAST "Nonce", BAD_CAST nonce_base64);
    xmlNewProp(nonce, BAD_CAST "EncodingType", BAD_CAST "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary");
    xmlNewTextChild(username_token, ns_wsu, BAD_CAST "Created", BAD_CAST time_holder);
}

void getBase64(unsigned char * buffer, int chunk_size, unsigned char * result) {
    char *c = (char *)result;
    int cnt = 0;
    base64_encodestate s;
    base64_init_encodestate(&s);
    cnt = base64_encode_block((char *)buffer, chunk_size, c, &s);
    c += cnt;
    cnt = base64_encode_blockend(c, &s);
    c += cnt;
    *c = 0;
}

void addHttpHeader(xmlDocPtr doc, xmlNodePtr root, char *xaddrs, char *post_type, char cmd[], int cmd_length) {
    xmlOutputBufferPtr outputbuffer = xmlAllocOutputBuffer(NULL);
    xmlNodeDumpOutput(outputbuffer, doc, root, 0, 0, NULL);
    int size = xmlOutputBufferGetSize(outputbuffer);

    char xml[8192] = {0};
    if (size > 8191) {
        fprintf(stderr, "xmlOutputBufferGetSize too big %d\n", size);
        strncat(xml, (char*)xmlOutputBufferGetContent(outputbuffer), 8191);
    } else {
        strcpy(xml, (char*)xmlOutputBufferGetContent(outputbuffer));
    }


    xmlOutputBufferFlush(outputbuffer);
    xmlOutputBufferClose(outputbuffer);
    xmlFreeDoc(doc);

    char c_xml_size[6];
    sprintf(c_xml_size, "%d", size);
    int xml_size_length = strlen(c_xml_size)+1;

    char tmp[128] = {0};
    char * mark = strstr(xaddrs, "//");
    int start = mark-xaddrs+2;
    int tmp_len = strlen(xaddrs);
    int j;
    for (j=0; j<tmp_len-start; j++) {
        if (j < 128)
            tmp[j] = xaddrs[j+start];
    }
    tmp[j] = '\0';

    mark = strstr(tmp, "/");
    int end = mark-tmp;
    char tmp2[128] = {0};
    for (j=0; j<end; j++) {
        tmp2[j] = tmp[j];
    }
    tmp2[j] = '\0';

    char host[128] = {0};
    char port_buf[128] = {0};
    mark = strstr(tmp2, ":");
    if (mark == NULL) {
        strcpy(host, tmp2);
        strcpy(port_buf, "80");
    } else {
        start = mark-tmp2;
        for (j=0; j<start; j++) {
            host[j] = tmp2[j];
        }
        host[j] = '\0';
        tmp_len = strlen(tmp2);
        for (j=start+1; j<tmp_len; j++) {
            port_buf[j-(start+1)] = tmp2[j];
        }
        port_buf[j-(start+1)] = '\0';
    }
    int port = atoi(port_buf);

    char content[] =
    "User-Agent: Generic\r\n"
    "Connection: Close\r\n"
    "Accept-Encoding: gzip, deflate\r\n"
    "Content-Type: application/soap+xml; charset=utf-8;\r\n"
    "Host: ";
    char content_length[] = "\r\nContent-Length: ";

    char http_terminate[5];
    http_terminate[0] = '\r';
    http_terminate[1] = '\n';
    http_terminate[2] = '\r';
    http_terminate[3] = '\n';
    http_terminate[4] = '\0';

    int p = strlen(post_type)+1;
    int h = strlen(host)+1;
    int c = sizeof(content);
    int cl = sizeof(content_length);
    int cmd_size = p + c + h + cl + xml_size_length + size + 1;
    int i;
    int s;
    for (i=0; i<p-1; i++)
        cmd[i] = post_type[i];
    s = i;
    for (i=0; i<c-1; i++)
        cmd[s+i] = content[i];
    s = s+i;
    for (i=0; i<h-1; i++)
        cmd[s+i] = host[i];
    s = s+i;
    for (i=0; i<cl-1; i++)
        cmd[s+i] = content_length[i];
    s = s+i;
    for (i=0; i<xml_size_length-1; i++)
        cmd[s+i] = c_xml_size[i];
    s = s+i;
    for (i=0; i<5-1; i++)
        cmd[s+i] = http_terminate[i];
    s = s+i;
    for (i=0; i<size; i++)
        cmd[s+i] = xml[i];
    cmd[cmd_size] = '\0';
}

void getUUID(char uuid_buf[47]) {
    srand(time(NULL));
    strcpy(uuid_buf, "urn:uuid:");
    for (int i=0; i<16; i++) {
        char buf[3];
        sprintf(buf, "%02x", (unsigned char) rand());
        strcat(uuid_buf, buf);
        if (i==3 || i==5 || i==7 || i==9)
            strcat(uuid_buf, "-");
    }
}

int broadcast(struct OnvifSession *onvif_session) {
    strcpy(preferred_network_address, onvif_session->preferred_network_address);
    struct sockaddr_in broadcast_address;
    int broadcast_socket;
    char broadcast_message[1024] = {0};
    unsigned int address_size;
    int error_code;

    if (onvif_session->discovery_msg_id == 1)
        getDiscoveryXml(broadcast_message, 1024, onvif_session->uuid);
    else if (onvif_session->discovery_msg_id == 2)
        getDiscoveryXml2(broadcast_message, 1024);

    int broadcast_message_length = strlen(broadcast_message);
    broadcast_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    setSocketOptions(broadcast_socket);
    for (int k=0; k<128; k++) {
        for (int j=0; j<8192; j++) {
            onvif_session->buf[k][j] = '\0';
        }
    }

    memset((char *) &broadcast_address, 0, sizeof(broadcast_address));
    broadcast_address.sin_family = AF_INET;
    broadcast_address.sin_port = htons(3702);
    broadcast_address.sin_addr.s_addr = inet_addr("239.255.255.250");
    int status = sendto(broadcast_socket, broadcast_message, broadcast_message_length, 0, (struct sockaddr*)&broadcast_address, sizeof(broadcast_address));
    if (status < 0) {
        //error
    }

    int i = 0;
    unsigned char looping = 1;
    address_size = sizeof(broadcast_address);
    while(looping) {
        onvif_session->len[i] = recvfrom(broadcast_socket, onvif_session->buf[i], sizeof(onvif_session->buf[i]), 0, (struct sockaddr*) &broadcast_address, &address_size);
        if (onvif_session->len[i] > 0) {
            onvif_session->buf[i][onvif_session->len[i]] = '\0';
            i++;
        } else {
            looping = 0;
            if (onvif_session->len[i] < 0) {
                //error
            }
        }
    }

#ifdef _WIN32
    closesocket(broadcast_socket);
#else
    close(broadcast_socket);
#endif

    return i;
}

void getActiveNetworkInterfaces(struct OnvifSession* onvif_session)
{
#ifdef _WIN32
    PIP_ADAPTER_INFO pAdapterInfo;
    PIP_ADAPTER_INFO pAdapter = NULL;
    DWORD dwRetVal = 0;
    int count = 0;

    ULONG ulOutBufLen = sizeof (IP_ADAPTER_INFO);
    pAdapterInfo = (IP_ADAPTER_INFO *) malloc(sizeof (IP_ADAPTER_INFO));
    if (pAdapterInfo == NULL) {
        printf("Error allocating memory needed to call GetAdaptersinfo\n");
        return;
    }

    if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);
        pAdapterInfo = (IP_ADAPTER_INFO *) malloc(ulOutBufLen);
        if (pAdapterInfo == NULL) {
            printf("Error allocating memory needed to call GetAdaptersinfo\n");
            return;
        }
    }

    if ((dwRetVal = GetAdaptersInfo(pAdapterInfo, &ulOutBufLen)) == NO_ERROR) {
        pAdapter = pAdapterInfo;
        while (pAdapter) {
            if (strcmp(pAdapter->IpAddressList.IpAddress.String, "0.0.0.0")) {
                char interface_info[1024];
                sprintf(interface_info, "%s - %s", pAdapter->IpAddressList.IpAddress.String, pAdapter->Description);
                printf("Network interface info %s\n", interface_info);
                //args.push_back(interface_info);
                strncpy(onvif_session->active_network_interfaces[count], interface_info, 40);
                count += 1;
            }
            pAdapter = pAdapter->Next;
        }
    } 
    else {
        printf("GetAdaptersInfo failed with error: %d", dwRetVal);
    }
    if (pAdapterInfo)
        free(pAdapterInfo);
#else
    struct ifaddrs *ifaddr;
    int family, s;
    char host[NI_MAXHOST];
    int count = 0;

    if (getifaddrs(&ifaddr) == -1) {
        printf("Error: getifaddrs failed - %s\n", strerror(errno));
        return;
    }

    for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;

        if (family == AF_INET ) {
            s = getnameinfo(ifa->ifa_addr, 
                    sizeof(struct sockaddr_in),
                    host, NI_MAXHOST,
                    NULL, 0, NI_NUMERICHOST);

            if (s != 0) {
                printf("getnameinfo() failed: %s\n", gai_strerror(s));
                continue;
            }

            if (strcmp(host, "127.0.0.1")) {
                strcpy(onvif_session->active_network_interfaces[count], host);
                strcat(onvif_session->active_network_interfaces[count], " - ");
                strcat(onvif_session->active_network_interfaces[count], ifa->ifa_name);
                count += 1;
            }
        } 
    }
    freeifaddrs(ifaddr);
#endif
}

void getIPAddress(char buf[128]) {
#ifdef _WIN32
    PMIB_IPADDRTABLE pIPAddrTable;
    DWORD dwSize = 0;
    DWORD dwRetVal = 0;
    IN_ADDR IPAddr;
    
    pIPAddrTable = (MIB_IPADDRTABLE *) malloc(sizeof(MIB_IPADDRTABLE));
    if (pIPAddrTable) {
        if (GetIpAddrTable(pIPAddrTable, &dwSize, 0) == ERROR_INSUFFICIENT_BUFFER) {
            free(pIPAddrTable);
            pIPAddrTable = (MIB_IPADDRTABLE *) malloc(dwSize);
        }
        if (pIPAddrTable == NULL) {
            return;
        }
    }

    if ((dwRetVal = GetIpAddrTable(pIPAddrTable, &dwSize, 0)) != NO_ERROR) {
        return;
    }

    int p = 0;
    while (p < (int)pIPAddrTable->dwNumEntries) {
        if (pIPAddrTable->table[p].dwAddr != inet_addr("127.0.0.1") && pIPAddrTable->table[p].dwMask == inet_addr("255.255.255.0")) {
            IPAddr.S_un.S_addr = (u_long)pIPAddrTable->table[p].dwAddr;
            strcpy(buf, inet_ntoa(IPAddr));
            p = (int)pIPAddrTable->dwNumEntries;
        }
        p++;
    }

    if (pIPAddrTable) {
        free(pIPAddrTable);
        pIPAddrTable = NULL;
    }

#else

#if defined(__APPLE__) || defined(__FreeBSD__)

    char *address;
    struct ifaddrs *interfaces = NULL;
    struct ifaddrs *temp_addr = NULL;
    int success = 0;
    success = getifaddrs(&interfaces);
    if (success == 0) {
        temp_addr = interfaces;
        while (temp_addr != NULL) {
            address = inet_ntoa(((struct sockaddr_in *)temp_addr->ifa_addr)->sin_addr);
            if (strcmp(address, "127.0.0.1") != 0)
                strcpy(buf, address);
        }
        temp_addr = temp_addr->ifa_next;
    }
    freeifaddrs(interfaces);

#else
    struct ifconf ifc;
    struct ifreq ifr[10];
    int sd, ifc_num, addr,mask, i;

    sd = socket(PF_INET, SOCK_DGRAM, 0);
    if (sd > 0) {
        ifc.ifc_len = sizeof(ifr);
        ifc.ifc_ifcu.ifcu_buf = (caddr_t)ifr;

        if (ioctl(sd, SIOCGIFCONF, &ifc) == 0) {
            ifc_num = ifc.ifc_len / sizeof(struct ifreq);

            for (i = 0; i < ifc_num; ++i) {
                if (ifr[i].ifr_addr.sa_family != AF_INET) {
                    continue;
                }

                if (ioctl(sd, SIOCGIFNETMASK, &ifr[i]) == 0) {
                    mask = ((struct sockaddr_in *)(&ifr[i].ifr_netmask))->sin_addr.s_addr;
                    char mask_buf[128] = {'/0'};
                    sprintf(mask_buf, "%d.%d.%d.%d", INT_TO_ADDR(mask));
                    if (strcmp(mask_buf, "255.255.255.0") == 0) {
                        if (ioctl(sd, SIOCGIFADDR, &ifr[i]) == 0) {
                            addr = ((struct sockaddr_in *)(&ifr[i].ifr_addr))->sin_addr.s_addr;
                            char addr_buf[128] = {'/0'};
                            sprintf(addr_buf, "%d.%d.%d.%d", INT_TO_ADDR(addr));
                            if (strcmp(addr_buf, "127.0.0.1") != 0) {
                                printf("-----------------------------------------------%s\n", addr_buf);
                                strcpy(buf, addr_buf);
                            }
                        }
                    }
                }
            }
        }
    }
    close(sd);
#endif /* not  __APPLE__ || __FreeBSD__ */
#endif /* not _WIN32 */
}

int mask2prefix(char *mask_buf) {
    struct in_addr mask;
    inet_pton(AF_INET, mask_buf, &mask);
    uint32_t number = ntohl(mask.s_addr);
    int count = 0;
    unsigned int step = 0;
    while (number > 0) {
        if (number & 1) {
            step = 1;
            count++;
        } else {
            if (step) {
                return -1;
            }
        }
        number >>=1;
    }
    return count;
}

void prefix2mask(int prefix, char mask_buf[128]) {
    struct in_addr mask;
    uint32_t number;

    if (prefix) {
        number = htonl(~((1 << (32-prefix)) - 1));
    } else {
        number = htonl(0);
    }

    mask.s_addr = number;
    inet_ntop(AF_INET, &mask, mask_buf, 128);
}

int setSocketOptions(int socket) {
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 500000;
    int broadcast = 500;
    char loopch = 0;
    int status = 0;
    struct in_addr localInterface;

#ifdef _WIN32
    PMIB_IPADDRTABLE pIPAddrTable;
    DWORD dwSize = 0;
    DWORD dwRetVal = 0;
    IN_ADDR IPAddr;

    pIPAddrTable = (MIB_IPADDRTABLE *) malloc(sizeof(MIB_IPADDRTABLE));
    if (pIPAddrTable) {
        if (GetIpAddrTable(pIPAddrTable, &dwSize, 0) == ERROR_INSUFFICIENT_BUFFER) {
            free(pIPAddrTable);
            pIPAddrTable = (MIB_IPADDRTABLE *) malloc(dwSize);
        }
        if (pIPAddrTable == NULL) {
            printf("Memory allocation failed for GetIpAddrTable\n");
            return -1;
        }
    }

    if ((dwRetVal = GetIpAddrTable(pIPAddrTable, &dwSize, 0)) != NO_ERROR) {
        printf("GetIpAddrTable failed with error %d\n", dwRetVal);
        return -1;
    }

    int p = 0;
    while (p < (int)pIPAddrTable->dwNumEntries) {
        IPAddr.S_un.S_addr = (u_long)pIPAddrTable->table[p].dwAddr;
        IPAddr.S_un.S_addr = (u_long)pIPAddrTable->table[p].dwMask;
        if (pIPAddrTable->table[p].dwAddr != inet_addr("127.0.0.1") && pIPAddrTable->table[p].dwMask == inet_addr("255.255.255.0")) {
            if (strlen(preferred_network_address) > 0) {
                localInterface.s_addr = inet_addr(preferred_network_address);
            }
            else {
                localInterface.s_addr = pIPAddrTable->table[p].dwAddr;
            }
            status = setsockopt(socket, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&localInterface, sizeof(localInterface));
            if (status < 0)
                printf("ip_multicast_if error");
            p = (int)pIPAddrTable->dwNumEntries;
        }
        p++;
    }

    if (pIPAddrTable) {
        free(pIPAddrTable);
        pIPAddrTable = NULL;
    }

    status = setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (const char *)&broadcast, sizeof(broadcast));
#else
    if (strlen(preferred_network_address) > 0) {
        localInterface.s_addr = inet_addr(preferred_network_address);
        status = setsockopt(socket, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&localInterface, sizeof(localInterface));
        if (status < 0)
            printf("ip_multicast_if error");
    }
    status = setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (struct timeval *)&tv, sizeof(struct timeval));
#endif
    status = setsockopt(socket, IPPROTO_IP, IP_MULTICAST_LOOP, (char *)&loopch, sizeof(loopch));
    return 0;
}



#ifdef __MINGW32__
int inet_pton(int af, const char *src, void *dst) {
    struct sockaddr_storage ss;
    int size = sizeof(ss);
    char src_copy[INET6_ADDRSTRLEN+1];

    ZeroMemory(&ss, sizeof(ss));
    strncpy (src_copy, src, INET6_ADDRSTRLEN+1);
    src_copy[INET6_ADDRSTRLEN] = 0;

    if (WSAStringToAddress(src_copy, af, NULL, (struct sockaddr *)&ss, &size) == 0) {
        switch(af) {
	case AF_INET:
	    *(struct in_addr *)dst = ((struct sockaddr_in *)&ss)->sin_addr;
	    return 1;
	case AF_INET6:
	    *(struct in6_addr *)dst = ((struct sockaddr_in6 *)&ss)->sin6_addr;
	    return 1;
	}
    }
    return 0;
}

const char *inet_ntop(int af, const void *src, char *dst, socklen_t size) {
    struct sockaddr_storage ss;
    unsigned long s = size;

    ZeroMemory(&ss, sizeof(ss));
    ss.ss_family = af;

    switch(af) {
    case AF_INET:
        ((struct sockaddr_in *)&ss)->sin_addr = *(struct in_addr *)src;
	break;
    case AF_INET6:
        ((struct sockaddr_in6 *)&ss)->sin6_addr = *(struct in6_addr *)src;
	break;
    default:
        return NULL;
    }

    return (WSAAddressToString((struct sockaddr *)&ss, sizeof(ss), NULL, dst, &s) == 0)?dst : NULL;
}
#endif


void extractOnvifService(char service[1024], bool post) {
    int length = strlen(service);
    char *sub = strstr(service, "//");
    if (sub != NULL) {
        int mark = sub - service;
        mark = mark+2;

        int i;
        for (i=0; i<length-mark; i++) {
            service[i] = service[i+mark];
        }
        service[i] = '\0';

        sub = strstr(service, " ");
        if (sub != NULL) {
            mark = sub - service;
            service[mark] = '\0';
        }

        length = strlen(service);
        sub = strstr(service, "/");
        if (sub != NULL) {
            mark = sub - service;
            for (i=0; i<length-mark; i++) {
                service[i] = service[i+mark];
            }
            service[i] = 0;

            if (post) {
                char temp_buf[128] = {0};
                strcat(temp_buf, "POST ");
                strcat(temp_buf, service);
                strcat(temp_buf, " HTTP/1.1\r\n");
                strcpy(service, "");
                strcpy(service, temp_buf);
            }
        }
    }
}


void extractHost(char *xaddrs, char host[128]) {
    char tmp[128] = {0};
    char *mark = strstr(xaddrs, "//");
    int start = mark-xaddrs+2;
    int tmp_len = strlen(xaddrs);
    int j;
    for (j=0; j<tmp_len-start; j++) {
        if (j < 128)
            tmp[j] = xaddrs[j+start];
    }
    tmp[j] = '\0';

    mark = strstr(tmp, "/");
    int end = mark-tmp;
    char tmp2[128] = {0};
    for (j=0; j<end; j++) {
        tmp2[j] = tmp[j];
    }
    tmp2[j] = '\0';

    mark = strstr(tmp2, ":");
    if (mark == NULL) {
        strcpy(host, tmp2);
    } else {
        start = mark-tmp2;
        for (j=0; j<start; j++) {
            host[j] = tmp2[j];
        }
        host[j] = '\0';
    }
}

void getScopeField(char *scope, char *field_name, char cleaned[1024]) {
    char *field;
    char field_contents[1024] = {0};
    char *mark;
    int length;
    char *result = NULL;

    field = strstr(scope, field_name);
    if (field != NULL) {
        field = field + strlen(field_name);
        mark = strstr(field, " ");
        if (mark != NULL) {
            length = mark - field;
            strncpy(field_contents, field, length);
        } else {
            strcpy(field_contents, field);
        }

        length = strlen(field_contents);
        int offset = 0;
        int j;
        for (int i=0; i<length; i++) {
            j = i - offset;
            if (field_contents[i] == '%') {
                char middle[3] = {0};
                i++; offset++;
                middle[0] = field_contents[i];
                i++; offset++;
                middle[1] = field_contents[i];
                char *ptr;
                int result = strtol(middle, &ptr, 16);
                cleaned[j] = result;
            } else {
                cleaned[j] = field_contents[i];
            }
        }
        cleaned[length] = '\0';
    }
}

void getCameraName(int ordinal, struct OnvifSession *onvif_session, struct OnvifData *onvif_data) {
    xmlDocPtr xml_input = xmlParseMemory(onvif_session->buf[ordinal], onvif_session->len[ordinal]);
    for(int i=0; i<1024; i++)
        onvif_data->camera_name[i] = '\0';

    char scopes[8192];
    getXmlValue(xml_input, BAD_CAST "//s:Body//d:ProbeMatches//d:ProbeMatch//d:Scopes", scopes, 8192);

    char temp_mfgr[1024] = {0};
    char temp_hdwr[1024] = {0};

    getScopeField(scopes, "onvif://www.onvif.org/name/", temp_mfgr);
    getScopeField(scopes, "onvif://www.onvif.org/hardware/", temp_hdwr);

    if (strlen(temp_mfgr) > 0) {
        strcat(onvif_data->camera_name, temp_mfgr);
    }
    if (strlen(temp_hdwr) > 0) {
        if (strstr(temp_mfgr, temp_hdwr) == NULL) {
            strcat(onvif_data->camera_name, " ");
            strcat(onvif_data->camera_name, temp_hdwr);
        }
    }

    if (strlen(onvif_data->camera_name)  == 0)
        strcpy(onvif_data->camera_name, "UNKNOWN CAMERA");

    xmlFreeDoc(xml_input);
}

bool extractXAddrs(int ordinal, struct OnvifSession *onvif_session, struct OnvifData *onvif_data) {
    bool result = false;
    xmlDocPtr xml_input = xmlParseMemory(onvif_session->buf[ordinal], onvif_session->len[ordinal]);
    if (getXmlValue(xml_input, BAD_CAST "//s:Body//d:ProbeMatches//d:ProbeMatch//d:XAddrs", onvif_data->xaddrs, 1024) == 0) {
        char *sub = strstr(onvif_data->xaddrs, " ");
        if (sub != NULL) {
            int mark = sub - onvif_data->xaddrs;
            char test[16] = {0};
            strncpy(test, onvif_data->xaddrs, 15);
            if (strcmp(test, "http://169.254.")) {
                onvif_data->xaddrs[mark] = '\0';
            }
            else {
                char other[128] = {0};
                if (strlen(sub) > 1) {
                    strcpy(other, sub+1);
                    memset(onvif_data->xaddrs, 0, 1024);
                    strcpy(onvif_data->xaddrs, other);
                }
            }
        }
        strcpy(onvif_data->device_service, onvif_data->xaddrs);
        result = true;
    }
    xmlFreeDoc(xml_input);
    return result;
}

void clearData(struct OnvifData *onvif_data) {
    for (int i=0; i<16; i++) {
        for (int j=0; j<128; j++) {
            onvif_data->resolutions_buf[i][j] = '\0';
        }
    }
    for (int i=0; i<3; i++) {
        for (int j=0; j<128; j++) {
            onvif_data->audio_encoders[i][j] = '\0';
        }
        for (int j=0; j<3; j++) {
            onvif_data->audio_sample_rates[i][j] = 0;
            onvif_data->audio_bitrates[i][j] = 0;
        }
    }
    for (int i=0; i<128; i++) {
        onvif_data->videoEncoderConfigurationToken[i] = '\0';
        onvif_data->networkInterfaceToken[i] = '\0';
        onvif_data->networkInterfaceName[i] = '\0';
        onvif_data->ip_address_buf[i] = '\0';
        onvif_data->default_gateway_buf[i] = '\0';
        onvif_data->dns_buf[i] = '\0';
        onvif_data->mask_buf[i] = '\0';
        onvif_data->videoSourceConfigurationToken[i] = '\0';
        onvif_data->video_encoder_name[i] = '\0';
        onvif_data->h264_profile[i] = '\0';
        onvif_data->multicast_address_type[i] = '\0';
        onvif_data->multicast_address[i] = '\0';
        onvif_data->session_time_out[i] = '\0';
        onvif_data->media_service[i] = '\0';
        onvif_data->imaging_service[i] = '\0';
        onvif_data->ptz_service[i] = '\0';
        onvif_data->event_service[i] = '\0';
        onvif_data->profileToken[i] = '\0';
        onvif_data->username[i] = '\0';
        onvif_data->password[i] = '\0';
        onvif_data->encoding[i] = '\0';
    	onvif_data->timezone[i] = '\0';
    	onvif_data->ntp_type[i] = '\0';
    	onvif_data->ntp_addr[i] = '\0';
        onvif_data->host[i] = '\0';
        onvif_data->serial_number[i] = '\0';
        onvif_data->audio_encoding[i] = '\0';
        onvif_data->audio_name[i] = '\0';
        onvif_data->audioEncoderConfigurationToken[i] = '\0';
        onvif_data->audioSourceConfigurationToken[i] = '\0';
        onvif_data->audio_session_timeout[i] = '\0';
        onvif_data->audio_multicast_type[i] = '\0';
        onvif_data->audio_multicast_address[i] = '\0';
    }
    for (int i=0; i<1024; i++) {
        onvif_data->xaddrs[i] = '\0';
        onvif_data->device_service[i] = '\0';
        onvif_data->stream_uri[i] = '\0';
        onvif_data->camera_name[i] = '\0';
        onvif_data->host_name[i] = '\0';
    }
    onvif_data->gov_length_min = 0;
    onvif_data->gov_length_max = 0;
    onvif_data->frame_rate_min = 0;
    onvif_data->frame_rate_max = 0;
    onvif_data->bitrate_min = 0;
    onvif_data->bitrate_max = 0;
    onvif_data->width = 0;
    onvif_data->height = 0;
    onvif_data->gov_length = 0;
    onvif_data->frame_rate = 0;
    onvif_data->bitrate = 0;
    onvif_data->use_count = 0;
    onvif_data->quality = 0;
    onvif_data->multicast_port = 0;
    onvif_data->multicast_ttl = 0;
    onvif_data->autostart = false;
    onvif_data->prefix_length = 0;
    onvif_data->dhcp_enabled = false;
    onvif_data->brightness_min = 0;
    onvif_data->brightness_max = 0;
    onvif_data->saturation_min = 0;
    onvif_data->saturation_max = 0;
    onvif_data->contrast_min = 0;
    onvif_data->contrast_max = 0;
    onvif_data->sharpness_min = 0;
    onvif_data->sharpness_max = 0;
    onvif_data->brightness = 0;
    onvif_data->saturation = 0;
    onvif_data->contrast = 0;
    onvif_data->sharpness = 0;
    onvif_data->time_offset = 0;
    onvif_data->event_listen_port = 0;
    onvif_data->guaranteed_frame_rate = false;
    onvif_data->encoding_interval = 0;
    onvif_data->datetimetype = '\0';
    onvif_data->dst = false;
    onvif_data->ntp_dhcp = false;
    onvif_data->audio_bitrate = 0;
    onvif_data->audio_sample_rate = 0;
    onvif_data->audio_use_count = 0;
    onvif_data->audio_multicast_port = 0;
    onvif_data->audio_multicast_TTL = 0;
    onvif_data->audio_multicast_auto_start = false;
    onvif_data->disable_video = false;
    onvif_data->analyze_video = false;
    onvif_data->disable_audio = false;
    onvif_data->analyze_audio = false;
    onvif_data->desired_aspect = 0;
    onvif_data->hidden = false;
    onvif_data->cache_max = 100;
    onvif_data->sync_audio = false;
}

void copyData(struct OnvifData *dst, struct OnvifData *src) {
    for (int i=0; i<16; i++) {
        for (int j=0; j<128; j++) {
            dst->resolutions_buf[i][j] = src->resolutions_buf[i][j];
        }
    }
    for (int i=0; i<3; i++) {
        for (int j=0; j<128; j++) {
            dst->audio_encoders[i][j] = src->audio_encoders[i][j];
        }
        for (int j=0; j<8; j++) {
            dst->audio_sample_rates[i][j] = src->audio_sample_rates[i][j];
            dst->audio_bitrates[i][j] = src->audio_bitrates[i][j];
        }
    }
    for (int i=0; i<128; i++) {
        dst->videoEncoderConfigurationToken[i] = src->videoEncoderConfigurationToken[i];
        dst->networkInterfaceToken[i] = src->networkInterfaceToken[i];
        dst->networkInterfaceName[i] = src->networkInterfaceName[i];
        dst->ip_address_buf[i] = src->ip_address_buf[i];
        dst->default_gateway_buf[i] = src->default_gateway_buf[i];
        dst->dns_buf[i] = src->dns_buf[i];
        dst->videoSourceConfigurationToken[i] = src->videoSourceConfigurationToken[i];
        dst->video_encoder_name[i] = src->video_encoder_name[i];
        dst->h264_profile[i] = src->h264_profile[i];
        dst->multicast_address_type[i] = src->multicast_address_type[i];
        dst->multicast_address[i] = src->multicast_address[i];
        dst->session_time_out[i] = src->session_time_out[i];
        dst->media_service[i] = src->media_service[i];
        dst->imaging_service[i] = src->imaging_service[i];
        dst->ptz_service[i] = src->ptz_service[i];
        dst->event_service[i] = src->event_service[i];
        dst->profileToken[i] = src->profileToken[i];
        dst->username[i] = src->username[i];
        dst->password[i] = src->password[i];
        dst->encoding[i] = src->encoding[i];
    	dst->timezone[i] = src->timezone[i];
    	dst->ntp_type[i] = src->ntp_type[i];
    	dst->ntp_addr[i] = src->ntp_addr[i];
        dst->host[i] = src->host[i];
        dst->serial_number[i] = src->serial_number[i];
        dst->audio_encoding[i] = src->audio_encoding[i];
        dst->audio_name[i] = src->audio_name[i];
        dst->audioEncoderConfigurationToken[i] = src->audioEncoderConfigurationToken[i];
        dst->audioSourceConfigurationToken[i] = src->audioSourceConfigurationToken[i];
        dst->audio_session_timeout[i] = src->audio_session_timeout[i];
        dst->audio_multicast_type[i] = src->audio_multicast_type[i];
        dst->audio_multicast_address[i] = src->audio_multicast_address[i];
    }
    for (int i=0; i<1024; i++) {
        dst->xaddrs[i] = src->xaddrs[i];
        dst->device_service[i] = src->device_service[i];
        dst->stream_uri[i] = src->stream_uri[i];
        dst->camera_name[i] = src->camera_name[i];
        dst->host_name[i] = src->host_name[i];
        dst->last_error[i] = src->last_error[i];
    }
    dst->gov_length_min = src->gov_length_min;
    dst->gov_length_max = src->gov_length_max;
    dst->frame_rate_min = src->frame_rate_min;
    dst->frame_rate_max = src->frame_rate_max;
    dst->bitrate_min = src->bitrate_min;
    dst->bitrate_max = src->bitrate_max;
    dst->width = src->width;
    dst->height = src->height;
    dst->gov_length = src->gov_length;
    dst->frame_rate = src->frame_rate;
    dst->bitrate = src->bitrate;
    dst->use_count = src->use_count;
    dst->quality = src->quality;
    dst->multicast_port = src->multicast_port;
    dst->multicast_ttl = src->multicast_ttl;
    dst->autostart = src->autostart;
    dst->prefix_length = src->prefix_length;
    dst->dhcp_enabled = src->dhcp_enabled;
    dst->brightness_min = src->brightness_min;
    dst->brightness_max = src->brightness_max;
    dst->saturation_min = src->saturation_min;
    dst->saturation_max = src->saturation_max;
    dst->contrast_min = src->contrast_min;
    dst->contrast_max = src->contrast_max;
    dst->sharpness_min = src->sharpness_min;
    dst->sharpness_max = src->sharpness_max;
    dst->brightness = src->brightness;
    dst->saturation = src->saturation;
    dst->contrast = src->contrast;
    dst->sharpness = src->sharpness;
    dst->time_offset = src->time_offset;
    dst->event_listen_port = src->event_listen_port;
    dst->guaranteed_frame_rate = src->guaranteed_frame_rate;
    dst->encoding_interval = src->encoding_interval;
    dst->datetimetype = src->datetimetype;
    dst->dst = src->dst;
    dst->ntp_dhcp = src->ntp_dhcp;
    dst->audio_bitrate = src->audio_bitrate;
    dst->audio_sample_rate = src->audio_sample_rate;
    dst->audio_use_count = src->audio_use_count;
    dst->audio_multicast_port = src->audio_multicast_port;
    dst->audio_multicast_TTL = src->audio_multicast_TTL;
    dst->audio_multicast_auto_start = src->audio_multicast_auto_start;
    dst->disable_video = src->disable_video;
    dst->analyze_video = src->analyze_video;
    dst->disable_audio = src->disable_audio;
    dst->analyze_audio = src->analyze_audio;
    dst->desired_aspect = src->desired_aspect;
    dst->hidden = src->hidden;
    dst->cache_max = src->cache_max;
    dst->sync_audio = src->sync_audio;
}

void initializeSession(struct OnvifSession *onvif_session) {
    getUUID(onvif_session->uuid);
    onvif_session->discovery_msg_id = 1;
    xmlInitParser ();
    for (int i=0; i<16; i++) {
        for (int j=0; j<1024; j++) {
            onvif_session->active_network_interfaces[i][j] = '\0';
        }
    }
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif
    strcpy(preferred_network_address, onvif_session->preferred_network_address);
}

void closeSession(struct OnvifSession *onvif_session) {
#ifdef _WIN32
    WSACleanup();
#endif
    xmlCleanupParser ();
}

bool prepareOnvifData(int ordinal, struct OnvifSession *onvif_session, struct OnvifData *onvif_data) {
    clearData(onvif_data);
    getCameraName(ordinal, onvif_session, onvif_data);
    if (!extractXAddrs(ordinal, onvif_session, onvif_data))
        return false;
    extractOnvifService(onvif_data->device_service, true);
    extractHost(onvif_data->xaddrs, onvif_data->host);
    getTimeOffset(onvif_data);
    return true;
}

int fillRTSPn(struct OnvifData *onvif_data, int profileIndex) {
    memset(onvif_data->last_error, 0, sizeof(onvif_data->last_error));
    int result = 0;
    result = getCapabilities(onvif_data);
    if (result == 0) {
        result = getProfileToken(onvif_data, profileIndex);
        if (result == 0) {
            result = getStreamUri(onvif_data);
        }
    }
    return result;
}

bool hasPTZ(struct OnvifData* onvif_data) {
    if (strcmp(onvif_data->ptz_service, "") == 0)
        return false;
    else
        return true;

}

void dumpXmlNode (xmlDocPtr doc, xmlNodePtr cur_node, char *prefix) {
    const char *name;
    const char *value;
    char new_prefix[1024];
    char attr[128];
    xmlAttrPtr prop;

    /* Traverse the tree */
    for (; cur_node; cur_node = cur_node->next) {
        if (cur_node->type == XML_ELEMENT_NODE) {
            name = (char *)(cur_node->name);
            value = (const char *)xmlNodeListGetString(doc, cur_node->xmlChildrenNode, 1);
            if (value) {
                printf("%s%s=%s\n", prefix ? prefix : "", name, value);
            } else {
                sprintf(new_prefix, "%s%s.", prefix ? prefix : "", name);
                for (prop = cur_node->properties; prop; prop = prop->next) {
                    if (prop->children && prop->children->content) {
                        printf("%s%s=%s\n", new_prefix, prop->name, prop->children->content);
                    }
                }
            }
        }
        dumpXmlNode(doc, cur_node->children, new_prefix);
    }
}

/* Dump xml document */
void dumpReply(xmlDocPtr reply) {
    if (reply != NULL) {
        xmlChar *xpath = BAD_CAST "//s:Body/*";
        xmlXPathObjectPtr body = getNodeSet(reply, xpath);
        if (body) {
            xmlNodeSetPtr nodeset = body->nodesetval;
            for (int i=0; i<nodeset->nodeNr; i++) {
                xmlNodePtr cur = nodeset->nodeTab[i];
                /* Skip error return */
                if (strcmp((char *)cur->name, "Fault") != 0) {
                    printf("[%s]\n", cur->name);
                    dumpXmlNode(reply, cur->children, NULL);
                }
            }
        }
    }
}

/* Dump all available onvif device configuration */
void dumpConfigAll (struct OnvifData *onvif_data) {
    xmlDocPtr reply;

    dump_reply = true;

    getNetworkInterfaces(onvif_data);
    getNetworkDefaultGateway(onvif_data);
    getDNS(onvif_data);
    getCapabilities(onvif_data);
    getVideoEncoderConfigurationOptions(onvif_data);
    getVideoEncoderConfiguration(onvif_data);
    getProfile(onvif_data);
    getOptions(onvif_data);
    getImagingSettings(onvif_data);
    getFirstProfileToken(onvif_data);
    getTimeOffset(onvif_data);
    getNTP(onvif_data);
    getHostname(onvif_data);
    getStreamUri(onvif_data);
    getDeviceInformation(onvif_data);

    dump_reply = false;
}
