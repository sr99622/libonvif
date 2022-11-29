/********************************************************************
* libavio/src/Encoder.cpp
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

#include "Encoder.h"

namespace avio
{

Encoder::Encoder(Writer& writer, AVMediaType mediaType) : mediaType(mediaType), writer(&writer)
{
    if (mediaType == AVMEDIA_TYPE_VIDEO) this->writer->videoEncoder = this;
    if (mediaType == AVMEDIA_TYPE_AUDIO) this->writer->audioEncoder = this;
}

void Encoder::init()
{
    const char* str = av_get_media_type_string(mediaType);
    strMediaType = (str ? str : "UNKOWN MEDIA TYPE");

    switch (mediaType) {
    case AVMEDIA_TYPE_VIDEO:
        openVideoStream();
        break;
    case AVMEDIA_TYPE_AUDIO:
        openAudioStream();
        break;
    default:
        ex.msg("Encoder constructor failed: unknown media type", MsgPriority::CRITICAL);
    }
}

Encoder::~Encoder()
{
    close();
}

void Encoder::close()
{
    if (enc_ctx)       avcodec_free_context(&enc_ctx);
    if (pkt)           av_packet_free(&pkt);
    if (hw_frame)      av_frame_free(&hw_frame);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
    if (cvt_frame)     av_frame_free(&cvt_frame);
    opened = false;
}

void Encoder::openVideoStream()
{
    if (!writer->fmt_ctx) 
        writer->init();

    AVFormatContext* fmt_ctx = writer->fmt_ctx;
    AVCodecID codec_id = writer->fmt_ctx->oformat->video_codec;

    first_pass = true;
    pts_offset = 0;

    try {
        const AVCodec* codec;
        ex.ck(pkt = av_packet_alloc(), CmdTag::APA);

        if (hw_device_type != AV_HWDEVICE_TYPE_NONE) {
            codec = avcodec_find_encoder_by_name(hw_video_codec_name.c_str());
            std::stringstream str;
            str << "avcodec_find_encoder_by_name: " << hw_video_codec_name;
            if (!codec) throw Exception(str.str());
        }
        else {
            codec = avcodec_find_encoder(fmt_ctx->oformat->video_codec);
            if (!codec) throw Exception("avcodec_find_encoder");
        }
        ex.msg(std::string("encoder opened codec ") + codec->long_name);

        ex.ck(stream = avformat_new_stream(fmt_ctx, NULL), CmdTag::ANS);
        stream->id = fmt_ctx->nb_streams - 1;
        writer->video_stream_id = stream->id;
        ex.ck(enc_ctx = avcodec_alloc_context3(codec), CmdTag::AAC3);

        enc_ctx->codec_id = codec_id;
        enc_ctx->bit_rate = video_bit_rate;
        enc_ctx->width = width;
        enc_ctx->height = height;
        stream->time_base = av_make_q(1, frame_rate);
        enc_ctx->time_base = stream->time_base;
        enc_ctx->gop_size = gop_size;

        cvt_frame = av_frame_alloc();
        cvt_frame->width = enc_ctx->width;
        cvt_frame->height = enc_ctx->height;
        cvt_frame->format = AV_PIX_FMT_YUV420P;
        av_frame_get_buffer(cvt_frame, 0);
        av_frame_make_writable(cvt_frame);

        if (hw_device_type != AV_HWDEVICE_TYPE_NONE) {
            enc_ctx->pix_fmt = hw_pix_fmt;
            if (!profile.empty())
                av_opt_set(enc_ctx->priv_data, "profile", profile.c_str(), 0);

            ex.ck(av_hwdevice_ctx_create(&hw_device_ctx, hw_device_type, NULL, NULL, 0), CmdTag::AHCC);
            ex.ck(hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx), CmdTag::AHCA);

            AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(hw_frames_ref->data);
            frames_ctx->format = hw_pix_fmt;
            frames_ctx->sw_format = sw_pix_fmt;
            frames_ctx->width = width;
            frames_ctx->height = height;
            frames_ctx->initial_pool_size = 20;

            ex.ck(av_hwframe_ctx_init(hw_frames_ref), CmdTag::AHCI);
            ex.ck(enc_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref), CmdTag::ABR);
            av_buffer_unref(&hw_frames_ref);

            ex.ck(hw_frame = av_frame_alloc(), CmdTag::AFA);
            ex.ck(av_hwframe_get_buffer(enc_ctx->hw_frames_ctx, hw_frame, 0), CmdTag::AHGB);
        }
        else {
            enc_ctx->pix_fmt = pix_fmt;
        }

        if (enc_ctx->codec_id == AV_CODEC_ID_MPEG2VIDEO) enc_ctx->max_b_frames = 2;
        if (enc_ctx->codec_id == AV_CODEC_ID_MPEG1VIDEO) enc_ctx->mb_decision = 2;

        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
            enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        ex.ck(avcodec_open2(enc_ctx, codec, &opts), CmdTag::AO2);
        ex.ck(avcodec_parameters_from_context(stream->codecpar, enc_ctx), CmdTag::APFC);

        std::stringstream str;
        str << "Encoder video stream width: " << enc_ctx->width << " height: " << enc_ctx->height;
        ex.msg(str.str());
        opened = true;
    }
    catch (const Exception& e) {
        ex.msg(e.what(), MsgPriority::CRITICAL, "Encoder video stream constructor exception: ");
        close();
    }
}

void Encoder::openAudioStream()
{
    if (!writer->fmt_ctx) 
        writer->init();

    AVFormatContext* fmt_ctx = writer->fmt_ctx;
    AVCodecID codec_id = writer->fmt_ctx->oformat->audio_codec;

    first_pass = true;
    pts_offset = 0;

    try {
        const AVCodec* codec;
        codec = avcodec_find_encoder(codec_id);

        if (codec)
            ex.msg(std::string("Encoder opened audio stream codec ") + codec->long_name);
        else
            throw Exception(std::string("avcodec_find_decoder could not find ") + avcodec_get_name(codec_id));

        ex.ck(pkt = av_packet_alloc(), CmdTag::APA);
        ex.ck(stream = avformat_new_stream(fmt_ctx, NULL), CmdTag::ANS);
        stream->id = fmt_ctx->nb_streams - 1;
        writer->audio_stream_id = stream->id;
        ex.ck(enc_ctx = avcodec_alloc_context3(codec), CmdTag::AAC3);

        enc_ctx->sample_fmt = sample_fmt;
        enc_ctx->bit_rate = audio_bit_rate;
        enc_ctx->sample_rate = sample_rate;
        enc_ctx->channel_layout = channel_layout;
        enc_ctx->channels = av_get_channel_layout_nb_channels(enc_ctx->channel_layout);
        enc_ctx->frame_size = nb_samples;
        stream->time_base = av_make_q(1, enc_ctx->sample_rate);

        cvt_frame = av_frame_alloc();
        cvt_frame->channels = enc_ctx->channels;
        cvt_frame->channel_layout = enc_ctx->channel_layout;
        cvt_frame->format = sample_fmt;
        cvt_frame->nb_samples = nb_samples;
        av_frame_get_buffer(cvt_frame, 0);
        av_frame_make_writable(cvt_frame);

        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
            enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        ex.ck(avcodec_open2(enc_ctx, codec, NULL), CmdTag::AO2);
        ex.ck(avcodec_parameters_from_context(stream->codecpar, enc_ctx), CmdTag::APFC);

        opened = true;
    }
    catch (const Exception& e) {
        ex.msg(e.what(), MsgPriority::CRITICAL, "AudioStream constructor exception: ");
        close();
    }
}

bool Encoder::cmpFrame(AVFrame* frame)
{
    if (mediaType == AVMEDIA_TYPE_VIDEO)
        return (frame->width == width && frame->height == height && frame->format == pix_fmt);
    if (mediaType == AVMEDIA_TYPE_AUDIO)
        return (frame->channels == channels && frame->channel_layout == channel_layout &&
            frame->nb_samples == nb_samples && frame->format == sample_fmt);

    return false;
}

int Encoder::encode(Frame& f)
{
    int ret = 0;

    try {

        if (!pkt_q) throw Exception("no packet queue");

        f.set_pts(stream);
        AVFrame* frame = f.m_frame;

        if (frame) {
            if (mediaType == AVMEDIA_TYPE_VIDEO && !cmpFrame(frame)) {
                if (!sws_ctx) {
                    ex.ck(sws_ctx = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                        enc_ctx->width, enc_ctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL), CmdTag::SGC);
                }
                ex.ck(sws_scale(sws_ctx, frame->data, frame->linesize, 0, enc_ctx->height,
                    cvt_frame->data, cvt_frame->linesize), CmdTag::SS);
                //cvt_frame->pts = frame->pts;
                cvt_frame->pts = av_rescale_q(frame->pts, video_time_base, enc_ctx->time_base);
                frame = cvt_frame;
                Frame qf(cvt_frame);
            }

            if (mediaType == AVMEDIA_TYPE_AUDIO && !cmpFrame(frame)) {
                if (!swr_ctx) {
                    ex.ck(swr_ctx = swr_alloc(), CmdTag::SA);
                    av_opt_set_int(swr_ctx, "in_channel_count", frame->channels, 0);
                    av_opt_set_int(swr_ctx, "out_channel_count", channels, 0);
                    av_opt_set_channel_layout(swr_ctx, "in_channel_layout", frame->channel_layout, 0);
                    av_opt_set_channel_layout(swr_ctx, "out_channel_layout", channel_layout, 0);
                    av_opt_set_int(swr_ctx, "in_sample_rate", frame->sample_rate, 0);
                    av_opt_set_int(swr_ctx, "out_sample_rate", sample_rate, 0);
                    av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", (AVSampleFormat)frame->format, 0);
                    av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", sample_fmt, 0);
                    ex.ck(swr_init(swr_ctx), CmdTag::SI);
                }
                ex.ck(swr_convert(swr_ctx, cvt_frame->data, cvt_frame->nb_samples, (const uint8_t**)frame->data, frame->nb_samples), CmdTag::SC);
                //cvt_frame->pts = frame->pts;
                cvt_frame->pts = av_rescale_q(total_samples, av_make_q(1, enc_ctx->sample_rate), enc_ctx->time_base);
                cvt_frame->sample_rate = sample_rate;
                total_samples += nb_samples;
                frame = cvt_frame;
                Frame qf(cvt_frame);
            }

            if (hw_device_type != AV_HWDEVICE_TYPE_NONE) {
                av_frame_copy_props(hw_frame, frame);
                ex.ck(av_hwframe_transfer_data(hw_frame, frame, 0), CmdTag::AHTD);
                ex.ck(avcodec_send_frame(enc_ctx, hw_frame), CmdTag::ASF);
            }
            else {
                ex.ck(avcodec_send_frame(enc_ctx, frame), "hw avcodec_send_frame");
            }
        }
        else {
            ex.ck(ret = avcodec_send_frame(enc_ctx, NULL), "avcodec_send_frame");
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(enc_ctx, pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0) 
                ex.ck(ret, CmdTag::ARP);

            pkt->stream_index = stream->index;

            AVPacket* tmp = av_packet_alloc();

            // needed to prevent corruption in dts
            if (pkt->dts > pkt->pts) 
                pkt->dts = pkt->pts - 2;

            tmp = av_packet_clone(pkt);

            if (first_pass) {
                pts_offset = tmp->pts;
                first_pass = false;
            }
            tmp->pts -= pts_offset;
            tmp->dts -= pts_offset;

            /*
            std::stringstream str;
            str
                << " index: " << tmp->stream_index
                << " flags: " << tmp->flags
                << " pts: " << tmp->pts
                << " dts: " << tmp->dts
                << " size: " << tmp->size
                << " duration: " << tmp->duration;

            std::cout << strMediaType << "  " << str.str() << std::endl;
            */

            writer->write(tmp);

            // are we missing av_packet_free?
        }
    }
    catch (const Exception& e) {
        if (!strcmp(e.what(), "End of file"))
            std::cout << strMediaType << "Encode::encode exception: " << e.what();
    }

    return ret;
}

}
