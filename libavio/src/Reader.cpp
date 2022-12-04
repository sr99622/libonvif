/********************************************************************
* libavio/src/Reader.cpp
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

extern "C"
{
#include <libavutil/channel_layout.h>
}

#include <filesystem>
#include <time.h>

#include "Reader.h"
#include "Decoder.h"
#include "Display.h"

#define MAX_TIMEOUT 10


time_t timeout_start = time(NULL);

static int interrupt_callback(void *ctx)
{
    avio::Reader* reader = (avio::Reader*)ctx;
    time_t diff = time(NULL) - timeout_start;

    if (diff > MAX_TIMEOUT || reader->request_break) {
        return 1;
    }
    return 0;
}

namespace avio {

Reader::Reader(const char* filename)
{
    //std::cout << "Reader opening " << filename << std::endl;
    AVDictionary* opts = NULL;
    av_dict_set(&opts, "stimeout", "10000000", 0);
    ex.ck(avformat_open_input(&fmt_ctx, filename, NULL, &opts), CmdTag::AOI);
    av_dict_free(&opts);
    timeout_start = time(NULL);

    AVIOInterruptCB cb = { interrupt_callback, this };
    fmt_ctx->interrupt_callback = cb;

    ex.ck(avformat_find_stream_info(fmt_ctx, NULL), CmdTag::AFSI);

    video_stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_stream_index < 0) 
        ex.msg("av_find_best_stream could not find video stream", MsgPriority::INFO);

    audio_stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (audio_stream_index < 0) 
        ex.msg("av_find_best_stream could not find audio stream", MsgPriority::INFO);

    std::filesystem::path path = filename;
    extension = path.extension().string();
    std::cout << "Reader successfully opened " << std::endl;
}

Reader::~Reader()
{
    avformat_close_input(&fmt_ctx);
}

AVPacket* Reader::read()
{
    if (closed)
        return NULL;

    int ret = 0;
    AVPacket* pkt = av_packet_alloc();

    try {
        if (!fmt_ctx) throw Exception("fmt_ctx null");
        timeout_start = time(NULL);
        ex.ck(ret = av_read_frame(fmt_ctx, pkt), CmdTag::ARF);
        timeout_start = time(NULL);
    }
    catch (const Exception& e) {
        if (ret != AVERROR_EOF) {
            ex.msg(e.what(), MsgPriority::CRITICAL, "Reader::read exception: ");
            if (ret == AVERROR_EXIT) {
                std::cout << "Camera connection timed out"  << std::endl;
                if (display) {
                    if (((Display*)display)->glWidget) {
                        ((Display*)display)->glWidget->emit cameraTimeout();
                    }
                }
            }
        }

        av_packet_free(&pkt);
        closed = true;
    }

    return pkt;
}

AVPacket* Reader::seek()
{
    int flags = AVSEEK_FLAG_FRAME;
    if (seek_target_pts < last_video_pts)
        flags |= AVSEEK_FLAG_BACKWARD;

    try {
        ex.ck(av_seek_frame(fmt_ctx, seek_stream_index(), seek_target_pts, flags), CmdTag::ASF);
    }
    catch (const Exception& e) {
        std::cout << e.what() << std::endl;
    }

    seek_target_pts = AV_NOPTS_VALUE;

    AVPacket* pkt = NULL;
    while (pkt = read()) {
        if (pkt->stream_index == seek_stream_index()) {
            seek_found_pts = pkt->pts;
            break;
        }
    }

    return pkt;
}

void Reader::request_seek(float pct)
{
    seek_target_pts = (start_time() + (pct * duration()) / av_q2d(fmt_ctx->streams[seek_stream_index()]->time_base)) / 1000;
}

bool Reader::seeking() 
{
    return seek_target_pts != AV_NOPTS_VALUE || seek_found_pts != AV_NOPTS_VALUE;
}

void Reader::start_from(int milliseconds)
{
    request_seek((start_time() + milliseconds) / (float)duration());
}

void Reader::end_at(int milliseconds)
{
    stop_play_at_pts = ((start_time() + milliseconds) / av_q2d(fmt_ctx->streams[seek_stream_index()]->time_base)) / 1000;
}

int Reader::seek_stream_index()
{
    return (video_stream_index >= 0 ? video_stream_index : audio_stream_index);
}

int64_t Reader::start_time()
{
    int64_t start_pts = (fmt_ctx->start_time == AV_NOPTS_VALUE ? 0 : fmt_ctx->start_time);
    return start_pts * AV_TIME_BASE / 1000000000;
}

int64_t Reader::duration()
{
    return fmt_ctx->duration * AV_TIME_BASE / 1000000000;
}

int64_t Reader::bit_rate()
{
    return fmt_ctx->bit_rate;
}

bool Reader::has_video()
{
    return video_stream_index >= 0;
}

int Reader::width()
{
    int result = -1;
    if (video_stream_index >= 0)
        result = fmt_ctx->streams[video_stream_index]->codecpar->width;
    return result;
}

int Reader::height()
{
    int result = -1;
    if (video_stream_index >= 0)
        result = fmt_ctx->streams[video_stream_index]->codecpar->height;
    return result;
}

AVRational Reader::frame_rate()
{
    AVRational result = av_make_q(0, 0);
    if (video_stream_index >= 0)
        result = fmt_ctx->streams[video_stream_index]->avg_frame_rate;
    return result;
}

AVPixelFormat Reader::pix_fmt()
{
    AVPixelFormat result = AV_PIX_FMT_NONE;
    if (video_stream_index >= 0)
        result = (AVPixelFormat)fmt_ctx->streams[video_stream_index]->codecpar->format;
    return result;
}

const char* Reader::str_pix_fmt()
{
    const char* result = "unkown pixel format";
    if (video_stream_index >= 0) {
        const char* name = av_get_pix_fmt_name((AVPixelFormat)fmt_ctx->streams[video_stream_index]->codecpar->format);
        if (name)
            result = name;
    }
    return result;
}

AVCodecID Reader::video_codec()
{
    AVCodecID result = AV_CODEC_ID_NONE;
    if (video_stream_index >= 0)
        result = fmt_ctx->streams[video_stream_index]->codecpar->codec_id;
    return result;
}

const char* Reader::str_video_codec()
{
    const char* result = "unkown codec";
    if (video_stream_index >= 0) {
        result = avcodec_get_name(fmt_ctx->streams[video_stream_index]->codecpar->codec_id);
    }
    return result;
}

int64_t Reader::video_bit_rate()
{
    int64_t result = -1;
    if (video_stream_index >= 0)
        result = fmt_ctx->streams[video_stream_index]->codecpar->bit_rate;
    return result;
}

AVRational Reader::video_time_base()
{
    AVRational result = av_make_q(0, 0);
    if (video_stream_index >= 0)
        result = fmt_ctx->streams[video_stream_index]->time_base;
    return result;
}

bool Reader::has_audio()
{
    return audio_stream_index >= 0;
}

int Reader::channels()
{
    int result = -1;
    if (audio_stream_index >= 0)
        result = fmt_ctx->streams[audio_stream_index]->codecpar->channels;
    return result;
}

int Reader::sample_rate()
{
    int result = -1;
    if (audio_stream_index >= 0)
        result = fmt_ctx->streams[audio_stream_index]->codecpar->sample_rate;
    return result;
}

int Reader::frame_size()
{
    int result = -1;
    if (audio_stream_index >= 0)
        result = fmt_ctx->streams[audio_stream_index]->codecpar->frame_size;
    return result;
}

uint64_t Reader::channel_layout()
{
    uint64_t result = 0;
    if (audio_stream_index >= 0) 
        result = fmt_ctx->streams[audio_stream_index]->codecpar->channel_layout;
    return result;
}

std::string Reader::str_channel_layout()
{
    char result[256] = { 0 };
    if (audio_stream_index >= 0) {
        uint64_t cl = fmt_ctx->streams[audio_stream_index]->codecpar->channel_layout;
        av_get_channel_layout_string(result, 256, channels(), cl);
    }

    return std::string(result);
}

AVSampleFormat Reader::sample_format()
{
    AVSampleFormat result = AV_SAMPLE_FMT_NONE;
    if (audio_stream_index >= 0)
        result = (AVSampleFormat)fmt_ctx->streams[video_stream_index]->codecpar->format;
    return result;
}

const char* Reader::str_sample_format()
{
    const char* result = "unkown sample format";
    if (audio_stream_index >= 0) {
        const char* name = av_get_sample_fmt_name((AVSampleFormat)fmt_ctx->streams[audio_stream_index]->codecpar->format);
        if (name)
            result = name;
    }
    return result;
}

AVCodecID Reader::audio_codec()
{
    AVCodecID result = AV_CODEC_ID_NONE;
    if (audio_stream_index >= 0)
        result = fmt_ctx->streams[audio_stream_index]->codecpar->codec_id;
    return result;
}

const char* Reader::str_audio_codec()
{
    const char* result = "unkown codec";
    if (audio_stream_index >= 0) {
        result = avcodec_get_name(fmt_ctx->streams[audio_stream_index]->codecpar->codec_id);
    }
    return result;
}

int64_t Reader::audio_bit_rate()
{
    int64_t result = -1;
    if (audio_stream_index >= 0)
        result = fmt_ctx->streams[audio_stream_index]->codecpar->bit_rate;
    return result;
}

AVRational Reader::audio_time_base()
{
    AVRational result = av_make_q(0, 0);
    if (audio_stream_index >= 0)
        result = fmt_ctx->streams[audio_stream_index]->time_base;
    return result;
}

std::string Reader::get_pipe_out_filename()
{
    std::string filename;

    if (pipe_out_filename.empty()) {
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::stringstream str;
        str << std::put_time(&tm, "%y%m%d%H%M%S");
        filename = str.str() + extension;
    }
    else {
        filename = pipe_out_filename;
    }

    if (!pipe_out_dir.empty())
        filename = pipe_out_dir + "/" + filename;

    return filename;
}

}


