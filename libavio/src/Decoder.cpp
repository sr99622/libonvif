/********************************************************************
* libavio/src/Decoder.cpp
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

#include "Decoder.h"

AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;

AVPixelFormat get_hw_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts)
{
    const AVPixelFormat* p;

    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

namespace avio
{

Decoder::Decoder(Reader& reader, AVMediaType mediaType, AVHWDeviceType hw_device_type) : reader(&reader), mediaType(mediaType)
{
    try {
        const char* str = av_get_media_type_string(mediaType);
        strMediaType = (str ? str : "UNKOWN MEDIA TYPE");

        stream_index = av_find_best_stream(reader.fmt_ctx, mediaType, -1, -1, NULL, 0);
        if (stream_index < 0) {
            std::stringstream str;
            str << "Error opening stream, unable to find " << strMediaType << " stream";
            throw Exception(str.str());
        }
        stream = reader.fmt_ctx->streams[stream_index];
        dec = avcodec_find_decoder(stream->codecpar->codec_id);

        if (!dec) {
            std::stringstream str;
            str << "avcodec_find_decoder could not find " << avcodec_get_name(stream->codecpar->codec_id);
            throw Exception(str.str());
        }

        ex.ck(dec_ctx = avcodec_alloc_context3(dec), AAC3);
        ex.ck(avcodec_parameters_to_context(dec_ctx, stream->codecpar), APTC);

        if (mediaType == AVMEDIA_TYPE_VIDEO && dec_ctx->pix_fmt != AV_PIX_FMT_YUV420P) {
            ex.ck(sws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
                dec_ctx->width, dec_ctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL), SGC);
            cvt_frame = av_frame_alloc();
            cvt_frame->width = dec_ctx->width;
            cvt_frame->height = dec_ctx->height;
            cvt_frame->format = AV_PIX_FMT_YUV420P;
            av_frame_get_buffer(cvt_frame, 0);
        }

        ex.ck(frame = av_frame_alloc(), AFA);
        if (hw_device_type != AV_HWDEVICE_TYPE_NONE) {
            ex.ck(sw_frame = av_frame_alloc(), AFA);
            for (int i = 0;; i++) {
                const AVCodecHWConfig* config;
                config = avcodec_get_hw_config(dec, i);

                if (!config) {
                    std::stringstream str;
                    str << strMediaType << " Decoder " << dec->name << " does not support device type " << av_hwdevice_get_type_name(hw_device_type);
                    throw Exception(str.str());
                }

                if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == hw_device_type) {
                    hw_pix_fmt = config->pix_fmt;
                    break;
                }
            }

            ex.ck(av_hwdevice_ctx_create(&hw_device_ctx, hw_device_type, NULL, NULL, 0), AHCC);
            dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
            dec_ctx->get_format = get_hw_format;
            const char* hw_pix_fmt_name;
            hw_pix_fmt_name = av_get_pix_fmt_name(hw_pix_fmt);
            ex.msg(hw_pix_fmt_name, MsgPriority::INFO, "using hw pix fmt: ");

            ex.ck(sws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, AV_PIX_FMT_NV12,
                dec_ctx->width, dec_ctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL), SGC);

            cvt_frame = av_frame_alloc();
            cvt_frame->width = dec_ctx->width;
            cvt_frame->height = dec_ctx->height;
            cvt_frame->format = AV_PIX_FMT_YUV420P;
            av_frame_get_buffer(cvt_frame, 0);
        }

        ex.ck(avcodec_open2(dec_ctx, dec, NULL), AO2);
    }
    catch (const Exception& e) {
        std::cout << strMediaType << " Decoder constructor exception: " << e.what() << std::endl;
        close();
    }
}

Decoder::~Decoder()
{
    close();
}

void Decoder::close()
{
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_frame_free(&cvt_frame);
    avcodec_free_context(&dec_ctx);
    av_buffer_unref(&hw_device_ctx);
    sws_freeContext(sws_ctx);
}

int Decoder::decode(AVPacket* pkt)
{
    int ret = 0;
    try 
    {
        if (!dec_ctx) throw Exception("dec_ctx null");

        ex.ck(ret = avcodec_send_packet(dec_ctx, pkt), ASP);

        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret < 0) {
                if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
                    return 0;
                }
                else if (ret < 0) {
                    ex.ck(ret, "error during decoding");
                }
            }

            Frame f;
            if (frame->format == hw_pix_fmt) {
                ex.ck(ret = av_hwframe_transfer_data(sw_frame, frame, 0), AHTD);
                ex.ck(av_frame_copy_props(sw_frame, frame));
                ex.ck(sws_scale(sws_ctx, sw_frame->data, sw_frame->linesize, 0, dec_ctx->height, 
                    cvt_frame->data, cvt_frame->linesize), SS);
                cvt_frame->pts = sw_frame->pts;
                f = Frame(cvt_frame);
            }
            else {
                if (sws_ctx) {
                    ex.ck(sws_scale(sws_ctx, frame->data, frame->linesize, 0, dec_ctx->height,
                        cvt_frame->data, cvt_frame->linesize), SS);
                    cvt_frame->pts = frame->pts;
                    f = Frame(cvt_frame);
                }
                else {
                    f = Frame(frame);
                }
            }

            f.set_rts(stream);
            if (show_frames) std::cout << strMediaType << " decoder " << f.description() << std::endl;
            frame_q->push(f);
        }
    }
    catch (const Exception& e) {
        std::cout << strMediaType << " Decoder::decode exception: " << e.what() << std::endl;
        ret = -1;
    }

    return ret;
}


}
