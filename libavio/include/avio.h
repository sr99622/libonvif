/********************************************************************
* libavio/include/avio.h
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

#ifndef AVIO_H
#define AVIO_H

#include "Exception.h"
#include "Queue.h"
#include "Reader.h"
#include "Decoder.h"
#include "Encoder.h"
#include "Writer.h"
#include "Pipe.h"
#include "Filter.h"
#include "Display.h"
#include "GLWidget.h"

namespace avio
{

static void show_pkt(AVPacket* pkt)
{
    std::stringstream str;
    str 
        << " index: " << pkt->stream_index
        << " flags: " << pkt->flags
        << " pts: " << pkt->pts
        << " dts: " << pkt->dts
        << " size: " << pkt->size
        << " duration: " << pkt->duration;

    std::cout << str.str() << std::endl;
}

static void read(Reader* reader, Queue<AVPacket*>* vpq, Queue<AVPacket*>* apq) 
{
    if (reader->vpq_max_size > 0 && vpq) vpq->set_max_size(reader->vpq_max_size);
    if (reader->apq_max_size > 0 && apq) apq->set_max_size(reader->apq_max_size);

    std::deque<AVPacket*> pkts;
    Pipe* pipe = nullptr;

    try {
        while (AVPacket* pkt = reader->read())
        {
            reader->running = true;
            if (reader->request_break) {
                if (vpq) {
                    while (vpq->size() > 0) {
                        AVPacket* tmp = vpq->pop();
                        av_packet_free(&tmp);
                    }
                }
                if (apq) {
                    while (apq->size() > 0) {
                        AVPacket* tmp = apq->pop();
                        av_packet_free(&tmp);
                    }
                }
                break;
            }

            if (reader->request_pipe_write) {
                if (!pipe) {
                    pipe = new Pipe(*reader);
                    std::string filename = reader->get_pipe_out_filename();
                    pipe->open(filename);
                    while (pkts.size() > 0) {
                        AVPacket* tmp = pkts.front();
                        pkts.pop_front();
                        pipe->write(tmp);
                        av_packet_free(&tmp);
                    }
                }
                AVPacket* tmp = av_packet_clone(pkt);
                pipe->write(tmp);
                av_packet_free(&tmp);
            }
            else {
                if (pipe) {
                    pipe->close();
                    delete pipe;
                    pipe = nullptr;
                }
                if (pkt->stream_index == reader->video_stream_index) {
                    if (pkt->flags) {
                        while (pkts.size() > 0) {
                            AVPacket* tmp = pkts.front();
                            pkts.pop_front();
                            av_packet_free(&tmp);
                        }
                    }
                }
                AVPacket* tmp = av_packet_clone(pkt);
                pkts.push_back(tmp);
            }

            if (reader->seek_target_pts != AV_NOPTS_VALUE) {
                av_packet_free(&pkt);
                pkt = reader->seek();
                if (!pkt) {
                    break;
                }
                if (vpq) while(vpq->size() > 0) vpq->pop();
                if (apq) while(apq->size() > 0) apq->pop();
            }

            if (reader->stop_play_at_pts != AV_NOPTS_VALUE && pkt->stream_index == reader->seek_stream_index()) {
                if (pkt->pts > reader->stop_play_at_pts) {
                    av_packet_free(&pkt);
                    break;
                }
            }

            if (pkt->stream_index == reader->video_stream_index) {
                if (reader->show_video_pkts) show_pkt(pkt);
                if (vpq) {
                    vpq->push(pkt);
                }
                else
                    av_packet_free(&pkt);
            }

            else if (pkt->stream_index == reader->audio_stream_index) {
                if (reader->show_audio_pkts) show_pkt(pkt);
                if (apq)
                    apq->push(pkt);
                else
                    av_packet_free(&pkt);
            }
        }
        if (vpq) vpq->push(NULL);
        if (apq) apq->push(NULL);
    }
    catch (const QueueClosedException& e) {}
    catch (const Exception& e) { std::cout << " reader failed: " << e.what() << std::endl; }
    reader->running = false;
}

static void decode(Decoder* decoder, Queue<AVPacket*>* pkt_q, Queue<Frame>* frame_q) 
{
    decoder->frame_q = frame_q;
    decoder->pkt_q = pkt_q;

    try {
        while (AVPacket* pkt = pkt_q->pop())
        {
            decoder->decode(pkt);
            av_packet_free(&pkt);
        }
        decoder->decode(NULL);
        decoder->frame_q->push(Frame(nullptr));
    }
    catch (const QueueClosedException& e) { }
    catch (const Exception& e) { 
        std::stringstream str;
        str << decoder->strMediaType << " decoder failed: " << e.what();
        std::cout << str.str() << std::endl;
        decoder->reader->exit_error_msg = str.str();
        //decoder->reader->request_break;
        decoder->decode(NULL);
        decoder->frame_q->push(Frame(nullptr));
    }
}

static void filter(Filter* filter, Queue<Frame>* q_in, Queue<Frame>* q_out)
{
    try {
        Frame f;
        filter->frame_out_q = q_out;
        while (true)
        {
            q_in->pop(f);
            filter->filter(f);
            if (!f.isValid())
                break;
        }
        filter->frame_out_q->push(Frame(nullptr));
    }
    catch (const QueueClosedException& e) {}
}

static void write(Writer* writer, Encoder* encoder)
{
    try {

        Frame f;
        while (true) 
        {
            encoder->frame_q->pop(f);
            if (encoder->show_frames) std::cout << f.description() << std::endl;
            if (writer->enabled) {

                if (!encoder->opened)
                    encoder->init();

                if (!writer->opened) {
                    std::string filename;
                    if (!writer->write_dir.empty())
                        filename = writer->write_dir + "/";

                    if (writer->filename.empty()) {
                        std::time_t t = std::time(nullptr);
                        std::tm tm = *std::localtime(&t);
                        std::stringstream str;
                        str << std::put_time(&tm, "%y%m%d%H%M%S");
                        filename += str.str() + "." + writer->m_format;
                    }
                    else {
                        filename += writer->filename;
                    }
                    writer->open(filename);
                }

                if (writer->opened && encoder->opened) encoder->encode(f);
            }
            else {

                if (writer->opened) {
                    if (encoder->opened) {
                        Frame tmp(nullptr);
                        encoder->encode(tmp);
                        encoder->close();
                    }
                    writer->close();
                }
            }
        }
    }
    catch (const QueueClosedException& e) 
    { 
        if (writer->opened) {
            if (encoder->opened) {
                Frame tmp(nullptr);
                encoder->encode(tmp);
                encoder->close();
            }
            writer->close();
        }
    }
}

static void pkt_drain(Queue<AVPacket*>* pkt_q) 
{
    try {
        while (AVPacket* pkt = pkt_q->pop())
        {
            av_packet_free(&pkt);
        }
    }
    catch (const QueueClosedException& e) {}
}

static void frame_drain(Queue<Frame>* frame_q) 
{
    Frame f;
    try {
        while (true) 
        {
            frame_q->pop(f);
            if (!f.isValid())
                break;
        }
    }
    catch (const QueueClosedException& e) {}
}

typedef std::map<std::string, Queue<AVPacket*>*> PKT_Q_MAP;
typedef std::map<std::string, Queue<Frame>*> FRAME_Q_MAP;

class Process
{

public:
    Reader*   reader       = nullptr;
    Decoder*  videoDecoder = nullptr;
    Decoder*  audioDecoder = nullptr;
    Filter*   videoFilter  = nullptr;
    Filter*   audioFilter  = nullptr;
    Writer*   writer       = nullptr;
    Encoder*  videoEncoder = nullptr;
    Encoder*  audioEncoder = nullptr;
    Display*  display      = nullptr;
    GLWidget* glWidget     = nullptr;

    PKT_Q_MAP pkt_queues;
    FRAME_Q_MAP frame_queues;
    std::vector<std::string> pkt_q_names;
    std::vector<std::string> frame_q_names;
    std::map<std::string, std::string> display_q_names;
    std::vector<std::string> frame_q_drain_names;
    std::vector<std::string> pkt_q_drain_names;
    std::vector<std::string> merge_filenames;
    
    int interleaved_q_size = 0;
    std::string interleaved_q_name;
    bool muxing = false;
    std::string mux_video_q_name;
    std::string mux_audio_q_name;

    std::vector<std::thread*> ops;

    void key_event(int keyCode)
    {
        SDL_Event event;
        event.type = SDL_KEYDOWN;
        event.key.keysym.sym = keyCode;
        SDL_PushEvent(&event);
    }

    void add_reader(Reader& reader_in)
    {
        reader = &reader_in;
        if (!reader_in.vpq_name.empty()) pkt_q_names.push_back(reader_in.vpq_name);
        if (!reader_in.apq_name.empty()) pkt_q_names.push_back(reader_in.apq_name);
    }

    void add_decoder(Decoder& decoder_in)
    {
        if (decoder_in.mediaType == AVMEDIA_TYPE_VIDEO)
            videoDecoder = &decoder_in;
        if (decoder_in.mediaType == AVMEDIA_TYPE_AUDIO)
            audioDecoder = &decoder_in;

        pkt_q_names.push_back(decoder_in.pkt_q_name);
        frame_q_names.push_back(decoder_in.frame_q_name);
    }

    void add_filter(Filter& filter_in)
    {
        if (filter_in.mediaType() == AVMEDIA_TYPE_VIDEO)
            videoFilter = &filter_in;
        if (filter_in.mediaType() == AVMEDIA_TYPE_AUDIO)
            audioFilter = &filter_in;

        frame_q_names.push_back(filter_in.q_in_name);
        frame_q_names.push_back(filter_in.q_out_name);
    }

    void add_encoder(Encoder& encoder_in)
    {
        if (encoder_in.mediaType == AVMEDIA_TYPE_VIDEO) {
            videoEncoder = &encoder_in;
            writer = videoEncoder->writer;
        }
        if (encoder_in.mediaType == AVMEDIA_TYPE_AUDIO) {
            audioEncoder = &encoder_in;
            writer = audioEncoder->writer;
        }
        pkt_q_names.push_back(encoder_in.pkt_q_name);
        frame_q_names.push_back(encoder_in.frame_q_name);
    }

    void add_frame_drain(const std::string& frame_q_name)
    {
        frame_q_drain_names.push_back(frame_q_name);
    }

    void add_packet_drain(const std::string& pkt_q_name)
    {
        pkt_q_drain_names.push_back(pkt_q_name);
    }

    void add_display(Display& display_in)
    {
        display = &display_in;

        if (!display->vfq_out_name.empty())
            frame_q_names.push_back(display->vfq_out_name);
        if (!display->afq_out_name.empty())
            frame_q_names.push_back(display->afq_out_name);
    }

    void add_widget(GLWidget* widget_in)
    {
        glWidget = widget_in;
        if (!display->vfq_out_name.empty())
            frame_q_names.push_back(display->vfq_out_name);
    }

    void cleanup()
    {
        //reader = nullptr;
        //videoDecoder = nullptr;
        //videoFilter = nullptr;
        //audioDecoder = nullptr;
        //display = nullptr;

        for (PKT_Q_MAP::iterator q = pkt_queues.begin(); q != pkt_queues.end(); ++q) {
            if (q->second) {
                while (q->second->size() > 0) {
                    AVPacket* pkt = q->second->pop();
                    av_packet_free(&pkt);
                }
                q->second->close();
            }
        }

        for (FRAME_Q_MAP::iterator q = frame_queues.begin(); q != frame_queues.end(); ++q) {
            if (q->second) {
                while (q->second->size() > 0) {
                    Frame f;
                    q->second->pop(f);
                }
            q->second->close();
            }
        }

        for (int i = 0; i < ops.size(); i++) {
            ops[i]->join();
            delete ops[i];
        }

        for (PKT_Q_MAP::iterator q = pkt_queues.begin(); q != pkt_queues.end(); ++q) {
            if (q->second)
                delete q->second;
        }

        for (FRAME_Q_MAP::iterator q = frame_queues.begin(); q != frame_queues.end(); ++q) {
            if (q->second)
                delete q->second;
        }
    }

    void run()
    {
        av_log_set_level(AV_LOG_PANIC);

        for (const std::string& name : pkt_q_names) {
            if (!name.empty()) {
                if (pkt_queues.find(name) == pkt_queues.end())
                    pkt_queues.insert({ name, new Queue<AVPacket*>() });
            }
        }

        for (const std::string& name : frame_q_names) {
            if (!name.empty()) {
                if (frame_queues.find(name) == frame_queues.end())
                    frame_queues.insert({ name, new Queue<Frame>() });
            }
        }

        if (reader) {
            ops.push_back(new std::thread(read, reader,
                reader->has_video() ? pkt_queues[reader->vpq_name] : nullptr, 
                reader->has_audio() ? pkt_queues[reader->apq_name] : nullptr));
        }

        if (videoDecoder) {
            ops.push_back(new std::thread(decode, videoDecoder,
                pkt_queues[videoDecoder->pkt_q_name], frame_queues[videoDecoder->frame_q_name]));
        }

        if (videoFilter) {
            ops.push_back(new std::thread(filter, videoFilter,
                frame_queues[videoFilter->q_in_name], frame_queues[videoFilter->q_out_name]));
        }

        if (glWidget) {
            if (!glWidget->vfq_in_name.empty()) glWidget->vfq_in = frame_queues[glWidget->vfq_in_name];
            if (!glWidget->vfq_out_name.empty()) glWidget->vfq_out = frame_queues[glWidget->vfq_out_name];
        }

        if (audioDecoder) {
            ops.push_back(new std::thread(decode, audioDecoder,
                pkt_queues[audioDecoder->pkt_q_name], frame_queues[audioDecoder->frame_q_name]));
        }

        if (audioFilter) {
            ops.push_back(new std::thread(filter, audioFilter,
                frame_queues[audioFilter->q_in_name], frame_queues[audioFilter->q_out_name]));
        }

        if (videoEncoder) {
            videoEncoder->pkt_q = pkt_queues[videoEncoder->pkt_q_name];
            videoEncoder->frame_q = frame_queues[videoEncoder->frame_q_name];
            if (videoEncoder->frame_q_max_size > 0) videoEncoder->frame_q->set_max_size(videoEncoder->frame_q_max_size);
            if (writer->enabled) videoEncoder->init();
            ops.push_back(new std::thread(write, videoEncoder->writer, videoEncoder));
        }

        if (audioEncoder) {
            audioEncoder->pkt_q = pkt_queues[audioEncoder->pkt_q_name];
            audioEncoder->frame_q = frame_queues[audioEncoder->frame_q_name];
            if (audioEncoder->frame_q_max_size > 0) audioEncoder->frame_q->set_max_size(audioEncoder->frame_q_max_size);
            if (writer->enabled) audioEncoder->init();
            ops.push_back(new std::thread(write, audioEncoder->writer, audioEncoder));
        }

        for (const std::string& name : frame_q_drain_names)
            ops.push_back(new std::thread(frame_drain, frame_queues[name]));

        for (const std::string& name : pkt_q_drain_names)
            ops.push_back(new std::thread(pkt_drain, pkt_queues[name]));

        if (display) {

            if (writer) display->writer = writer;
            if (audioDecoder) display->audioDecoder = audioDecoder;
            if (audioFilter) display->audioFilter = audioFilter;

            if (!display->vfq_in_name.empty()) display->vfq_in = frame_queues[display->vfq_in_name];
            if (!display->afq_in_name.empty()) display->afq_in = frame_queues[display->afq_in_name];

            if (!display->vfq_out_name.empty()) display->vfq_out = frame_queues[display->vfq_out_name];
            if (!display->afq_out_name.empty()) display->afq_out = frame_queues[display->afq_out_name];

            display->init();

            if (glWidget)
                glWidget->emit timerStart();

            while (display->display()) {}

            if (glWidget)
                glWidget->emit timerStop();

            int count = 0;
            if (!reader->exit_error_msg.empty()) {
                reader->request_break = true;
                while (reader->running) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    std::cout << "reader running: " << count << std::endl;
                    if (count++ > 1000) {
                        if (!reader->apq_name.empty()) {
                            Queue<AVPacket*>* q = pkt_queues[reader->apq_name];
                            if (q) {
                                while (q->size() > 0) {
                                    AVPacket* pkt = q->pop();
                                    av_packet_free(&pkt);
                                }
                            }
                        }
                        if (!reader->vpq_name.empty()) {
                            Queue<AVPacket*>* q = pkt_queues[reader->vpq_name];
                            if (q) {
                                while (q->size() > 0) {
                                    AVPacket* pkt = q->pop();
                                    av_packet_free(&pkt);
                                }
                            }
                        }
                        break;
                    }
                }
                throw Exception(reader->exit_error_msg);
            }

            if (writer) {
                while (!display->audio_eof)
                    SDL_Delay(1);
                writer->enabled = false;
            }

        }

        cleanup();
    }

};

}

#endif // AVIO_H