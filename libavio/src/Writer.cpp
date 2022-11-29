/********************************************************************
* libavio/src/Writer.cpp
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

#include "Writer.h"
#include "Encoder.h"

namespace avio
{

Writer::Writer(const std::string& format) : m_format(format)
{

}

Writer::~Writer()
{

}

void Writer::init()
{
    ex.ck(avformat_alloc_output_context2(&fmt_ctx, NULL, m_format.c_str(), NULL), AAOC2);
}

void Writer::open(const std::string& filename)
{
    std::unique_lock<std::mutex> lock(mutex);
    try {
        if (getEncoderState() == EncoderState::OPEN && !opened) {
            if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE))
                ex.ck(avio_open(&fmt_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE), AO);

            ex.ck(avformat_write_header(fmt_ctx, NULL), AWH);
            opened = true;
        }
    }
    catch (const Exception& e) {
        std::cout << "Writer::open exception: " << e.what() << std::endl;
    }
}

void Writer::write(AVPacket* pkt)
{
    std::unique_lock<std::mutex> lock(mutex);
    try {
        ex.ck(av_interleaved_write_frame(fmt_ctx, pkt), AIWF);
    }
    catch (const Exception& e) {
        std::cout << "Writer::write exception: " << e.what() << std::endl;
    }
}

void Writer::close()
{
    std::unique_lock<std::mutex> lock(mutex);
    try {
        if (getEncoderState() == EncoderState::CLOSED && opened) {
            ex.ck(av_write_trailer(fmt_ctx), AWT);

            if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE))
                ex.ck(avio_closep(&fmt_ctx->pb), AC);

            avformat_free_context(fmt_ctx);
            fmt_ctx = NULL;

            opened = false;
        }
    }
    catch (const Exception& e) {
        std::cout << "Writer::close exception: " << e.what() << std::endl;
    }
}

EncoderState Writer::getEncoderState()
{
    bool videoEncoderOpen = false;
    bool videoEncoderClosed = false;
    if (videoEncoder) {
        if (((Encoder*)videoEncoder)->opened)
            videoEncoderOpen = true;
        else
            videoEncoderClosed = true;
    }

    bool audioEncoderOpen = false;
    bool audioEncoderClosed = false;
    if (audioEncoder) {
        if (((Encoder*)audioEncoder)->opened)
            audioEncoderOpen = true;
        else
            audioEncoderClosed = true;
    }

    if (videoEncoderOpen && audioEncoderOpen)
        return EncoderState::OPEN;
    if (videoEncoderOpen && !audioEncoder)
        return EncoderState::OPEN;
    if (!videoEncoder && audioEncoderOpen)
        return EncoderState::OPEN;

    if (videoEncoderClosed && audioEncoderClosed)
        return EncoderState::CLOSED;
    if (videoEncoderClosed && !audioEncoder)
        return EncoderState::CLOSED;
    if (!videoEncoder && audioEncoderClosed)
        return EncoderState::CLOSED;

    return EncoderState::MIXED;

}

}
