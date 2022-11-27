/********************************************************************
* libavio/include/Writer.h
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

#ifndef WRITER_H
#define WRITER_H

extern "C" {
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

#include <mutex>

#include "Exception.h"

namespace avio
{

enum class EncoderState {
    MIXED,
    OPEN,
    CLOSED
};

class Writer
{
public:
    Writer(const std::string& format);
    ~Writer();
    void open(const std::string& filename);
    void write(AVPacket* pkt);
    void close();
    void init();
    EncoderState getEncoderState();

    AVFormatContext* fmt_ctx = NULL;
    int video_stream_id = AVERROR_STREAM_NOT_FOUND;
    int audio_stream_id = AVERROR_STREAM_NOT_FOUND;

    std::string m_format;
    std::string write_dir;
    std::string filename;

    void* videoEncoder = nullptr;
    void* audioEncoder = nullptr;
    bool enabled = false;
    bool opened = false;

    bool show_video_pkts = false;
    bool show_audio_pkts = false;

    std::mutex mutex;

    ExceptionHandler ex;

};

}

#endif // WRITER_H