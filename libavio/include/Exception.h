/********************************************************************
* libavio/include/Exception.h
*
* Copyright (c) 2022  Stephen Rhodes
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*********************************************************************/

#pragma warning(disable: 26812)  // unscoped enum 

#ifndef EXCEPTION_H
#define EXCEPTION_H

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
}

#include <exception>
#include <iostream>
#include <functional>
#include <sstream>


    enum CmdTag {
        NONE,
        AO2,
        AOI,
        ACI,
        AFSI,
        APTC,
        APFC,
        AWH,
        AWT,
        AO,
        AC,
        ACP,
        AAOC2,
        AFMW,
        AFGB,
        AHCC,
        AFBS,
        AWF,
        ASP,
        ASF,
        AEV2,
        ARF,
        ADV2,
        ARP,
        AIWF,
        AFE,
        AFD,
        AAC3,
        AFA,
        AAC,
        AFC,
        ABR,
        AGF,
        AGA,
        AGC,
        AL,
        AGPP,
        AGCF,
        AHCA,
        AHCI,
        AHGB,
        AFEBN,
        AICTB,
        AGPFN,
        ABAFF,
        APFDG,
        AHFTBN,
        AOSI,
        AOSIL,
        AFDBN,
        ACLFM,
        ACLD,
        AGHC,
        AHTD,
        ANS,
        AFCP,
        SGC,
        AFIF,
        APA,
        ADC,
        AIA,
        AFR,
        AM,
        SASO,
        SA,
        SI,
        SC,
        SS
    };

    enum MsgPriority {
        CRITICAL,
        DEBUG,
        INFO
    };

namespace avio
 {
        class Exception : public std::exception
    {
    public:
        Exception(const char* msg);
        Exception(const std::string& msg);
        ~Exception();

        const char* what() const throw () {
            return buffer;
        }

        char* tmp = nullptr;
        const char* buffer;
    };

    class ExceptionHandler
    {

    public:
        void ck(int ret);
        void ck(int ret, CmdTag cmd_tag);
        void ck(int ret, const std::string& msg);
        void ck(void* arg, CmdTag cmd_tag = CmdTag::NONE);

        const Exception getNullException(CmdTag cmd_tag);
        const char* tag(CmdTag cmd_tag);

        std::function<void(const std::string&, MsgPriority, const std::string&)> fnMsgOut = nullptr;
        void msg(const std::string& msg, MsgPriority priority = MsgPriority::INFO, const std::string& qualifier = "");
    };

}


#endif // EXCEPTION_H
