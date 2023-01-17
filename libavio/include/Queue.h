/********************************************************************
* libavio/include/Queue.h
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

#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <exception>
#include <functional>

namespace avio
{

class QueueClosedException : public std::exception {
public:
	const char* what() const throw() {
		return "attempt to access closed queue";
	}
};

template <typename T>
class Queue
{
public:
	Queue(size_t max_size = 1);

	void push(T const&);
	void push_move(T);
	T pop();
	void pop(T&);
	void pop_move(T&);
	T peek();
	int size();
	void close();
	bool full() { return m_size == m_max_size; }
	bool empty() { return m_front == -1; }
	bool closed() { return m_closed; }
	void clear() { m_front = m_rear = -1; }
	void set_max_size(int arg);

private:
	std::vector<T> m_data;
	int m_max_size;
	int m_front = -1;
	int m_rear = -1;
	std::mutex n_mutex;
	std::condition_variable m_cond_push, m_cond_pop;
	bool m_closed = false;
	int m_size = 0;

};

template <typename T>
Queue<T>::Queue(size_t max_size) :	m_max_size(max_size) 
{
	m_data.reserve(max_size);
}

template <typename T>
void Queue<T>::set_max_size(int arg)
{
	m_max_size = arg;
	m_data.reserve(m_max_size);
}

template <typename T>
void Queue<T>::push(T const& element)
{
	std::unique_lock<std::mutex> lock(n_mutex);

	while (full()) {
		if (m_closed) break;
		m_cond_push.wait(lock);
	}

	if (m_closed) throw QueueClosedException();

	if (m_front == -1) m_front = m_rear = 0;
	else if (m_rear == m_max_size - 1 && m_front != 0) m_rear = 0;
	else m_rear++;

	if (m_data.size() < m_rear + 1)	m_data.push_back(element);
	else m_data[m_rear] = element;
	m_size++;

	m_cond_pop.notify_one();
}

template <typename T>
void Queue<T>::push_move(T element)
{
	std::unique_lock<std::mutex> lock(n_mutex);

	while (full()) {
		if (m_closed) break;
		m_cond_push.wait(lock);
	}

	if (m_closed) throw QueueClosedException();

	if (m_front == -1) m_front = m_rear = 0;
	else if (m_rear == m_max_size - 1 && m_front != 0) m_rear = 0;
	else m_rear++;

	if (m_data.size() < m_rear + 1)	{
		m_data.push_back(std::move(element));
	}
	else {
		m_data[m_rear] = std::move(element);
	}


	m_size++;

	m_cond_pop.notify_one();
}

template <typename T>
T Queue<T>::pop()
{
	std::unique_lock<std::mutex> lock(n_mutex);

	while (empty()) {
		if (m_closed) break;
		m_cond_pop.wait(lock);
	}

	if (m_closed) throw QueueClosedException();

	T& result = m_data[m_front];
	if (m_front == m_rear) m_front = m_rear = -1;
	else if (m_front == m_max_size - 1) m_front = 0;
	else m_front++;
	m_size--;

	m_cond_push.notify_one();
	return result;
}

template <typename T>
void Queue<T>::pop(T& arg)
{
	std::unique_lock<std::mutex> lock(n_mutex);

	while (empty()) {
		if (m_closed) break;
		m_cond_pop.wait(lock);
	}

	if (m_closed) throw QueueClosedException();

	arg = m_data[m_front];
	if (m_front == m_rear) m_front = m_rear = -1;
	else if (m_front == m_max_size - 1) m_front = 0;
	else m_front++;
	m_size--;

	m_cond_push.notify_one();
}

template <typename T>
void Queue<T>::pop_move(T& arg)
{
	std::unique_lock<std::mutex> lock(n_mutex);

	while (empty()) {
		if (m_closed) break;
		m_cond_pop.wait(lock);
	}

	if (m_closed) throw QueueClosedException();

	arg = std::move(m_data[m_front]);
	if (m_front == m_rear) m_front = m_rear = -1;
	else if (m_front == m_max_size - 1) m_front = 0;
	else m_front++;
	m_size--;

	m_cond_push.notify_one();
}

template <typename T>
T Queue<T>::peek()
{
	std::unique_lock<std::mutex> lock(n_mutex);

	while (empty()) {
		if (m_closed) break;
		m_cond_pop.wait(lock);
	}

	if (m_closed) throw QueueClosedException();

	T result = T(m_data[m_front]);
	return result;
}

template <typename T>
int Queue<T>::size()
{
	std::lock_guard<std::mutex> lock(n_mutex);
	return m_size;
}

template <typename T>
void Queue<T>::close()
{
	std::unique_lock<std::mutex> lock(n_mutex);
	m_closed = true;
	m_cond_push.notify_all();
	m_cond_pop.notify_all();
}

}

#endif // QUEUE_H