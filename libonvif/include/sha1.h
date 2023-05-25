/*
   SHA-1 in C
   By Steve Reid <steve@edmweb.com>
   100% Public Domain

The person or persons who have associated work with this document (the "Dedicator" or "Certifier")
hereby either (a) certifies that, to the best of his knowledge, the work of authorship identified 
is in the public domain of the country from which the work is published, or (b) hereby dedicates 
whatever copyright the dedicators holds in the work of authorship identified below (the "Work") to 
the public domain. A certifier, moreover, dedicates any copyright interest he may have in the 
associated work, and for these purposes, is described as a "dedicator" below.

A certifier has taken reasonable steps to verify the copyright status of this work. Certifier 
recognizes that his good faith efforts may not shield him from liability if in fact the work certified 
is not in the public domain.

Dedicator makes this dedication for the benefit of the public at large and to the detriment of the 
Dedicator's heirs and successors. Dedicator intends this dedication to be an overt act of relinquishment 
in perpetuity of all present and future rights under copyright law, whether vested or contingent, in the 
Work. Dedicator understands that such relinquishment of all rights includes the relinquishment of all 
rights to enforce (by lawsuit or otherwise) those copyrights in the Work.

Dedicator recognizes that, once placed in the public domain, the Work may be freely reproduced, distributed, 
transmitted, used, modified, built upon, or otherwise exploited by anyone for any purpose, commercial or 
non-commercial, and in any way, including by methods that have not yet been invented or conceived.

CC0 for Public Domain Dedication
This tool is based on United States law and may not be applicable outside the US. For dedicating new works 
to the public domain, we recommend CC0.
 */

#ifndef SHA1_H
#define SHA1_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"

typedef struct
{
    uint32_t state[5];
    uint32_t count[2];
    unsigned char buffer[64];
} SHA1_CTX;

void SHA1Transform(
    uint32_t state[5],
    const unsigned char buffer[64]
    );

void SHA1Init(
    SHA1_CTX * context
    );

void SHA1Update(
    SHA1_CTX * context,
    const unsigned char *data,
    uint32_t len
    );

void SHA1Final(
    unsigned char digest[20],
    SHA1_CTX * context
    );

void SHA1(
    char *hash_out,
    const char *str,
    int len);

#ifdef __cplusplus
}
#endif

#endif /* SHA1_H */
