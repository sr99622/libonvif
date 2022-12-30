/*******************************************************************************
* main.cpp
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL AV_ARRAY_API
#include "numpy/arrayobject.h"
#include "pyhelper.h"
#define NO_IMPORT_ARRAY

#include <functional>
#include <iostream>

#include "mainwindow.h"
#include <QApplication>

#include "Frame.h"
#include "Exception.h"

namespace avio
{

class PyRunner {
public:
	PyRunner() {}
	int initialize(const std::string& dir, const std::string& file, const std::string& py_class, const std::string& arg);
	CPyObject getImage(Frame& f);
	bool run(Frame& f, const std::string& events);
    void test(const std::string& arg1, const std::string& arg2);

	CPyObject pClass = nullptr;
	Queue<Frame>* frame_q = nullptr;
	ExceptionHandler ex;
	bool initialized = false;
	bool loop_back = false;

	PyObject* pData = NULL;

};

int PyRunner::initialize(const std::string& dir, const std::string& file, const std::string& py_class, const std::string& arg)
{
	std::cout << "python_dir: " << dir << std::endl;
	std::cout << "python_file: " << file << std::endl;
	std::cout << "python_class: " << py_class << std::endl;
	std::cout << "init arg: " << arg << std::endl;

    //import_array();
    //if (PyErr_Occurred())
    //    std::cout << "you're fucked" << std::endl;
    


    try {
        //if (pClass) delete pClass;

		CPyObject sysPath = PySys_GetObject("path");
		CPyObject dirName = PyUnicode_FromString(dir.c_str());
		PyList_Append(sysPath, dirName);

        PyRun_SimpleString("import sys");
        PyRun_SimpleString("print('this is a test', sys.path)");
        //PyRun_SimpleString("import echo");

		CPyObject pName = PyUnicode_FromString(file.c_str());		     if (!pName)   throw Exception("pName");
		CPyObject pModule = PyImport_Import(pName);                      if (!pModule) throw Exception("pModule");
		CPyObject pDict = PyModule_GetDict(pModule);                     if (!pDict)   throw Exception("pDict");
		CPyObject pItem = PyDict_GetItemString(pDict, py_class.c_str()); if (!pItem)   throw Exception("pItem");

		CPyObject pWrap = NULL;
		CPyObject pArg = NULL;

		if (arg.length() > 0) {
			pArg = Py_BuildValue("(s)", arg.c_str());
			pWrap = PyTuple_New(1);
			PyTuple_SetItem(pWrap, 0, pArg);
		}

		pClass = PyObject_CallObject(pItem, pWrap);                      if (!pClass) throw Exception("pClass");
		std::cout << "PyRunner construction complete" << std::endl;
	}
	catch (const Exception& e) {
		std::cout << "PyRunner constuctor exception: " << e.what() << std::endl;
		//std::exit(0);
	}

    return 0;
}

CPyObject PyRunner::getImage(Frame& f)
{
	// Note that this conversion will depend on the scope of Frame f to preserve data in mat_buf

	CPyObject result;
	try {
		if (!PyArray_API) throw Exception("numpy not initialized");
		int depth = -1;
		switch (f.m_frame->format) {
			case AV_PIX_FMT_BGR24:
			case AV_PIX_FMT_RGB24:
				depth = 3;
				break;
			case AV_PIX_FMT_BGRA:
			case AV_PIX_FMT_RGBA:
				depth = 4;
				break;
			default:
				throw Exception("unsupported pix fmt for numpy conversion, use video filter to convert format to bgr24, rgb24, bgra or rgba");
		}

		int width = f.m_frame->width;
		int height = f.m_frame->height;
		int linesize = width * depth;
		if (f.mat_buf) delete[] f.mat_buf;
		f.mat_buf = new uint8_t[linesize * height];

		for (int y = 0; y < height; y++)
			memcpy(f.mat_buf + y * linesize, f.m_frame->data[0] + y * f.m_frame->linesize[0], linesize);

		npy_intp dimensions[3] = { f.m_frame->height, f.m_frame->width, depth };
		pData = PyArray_SimpleNewFromData(3, dimensions, NPY_UINT8, f.mat_buf);
		if (!pData) throw Exception("pData");
		result = Py_BuildValue("(O)", pData);
	}
	catch (const Exception& e) {
		ex.msg(e.what(), MsgPriority::CRITICAL, "PyRunner::getImage exception: ");
		std::exit(0);
	}
	return result;
}

bool PyRunner::run(Frame& f, const std::string& events)
{
	bool result = false;
	if (f.isValid()) {
		CPyObject pImage = getImage(f);
		CPyObject pRTS = Py_BuildValue("(i)", f.m_rts);

		// pts is long long so use stream writer to convert
		std::stringstream str;
		str << f.m_frame->pts;
		CPyObject pStrPTS = Py_BuildValue("(s)", str.str().c_str());

		CPyObject pEvents = Py_BuildValue("(s)", events.c_str());

		CPyObject pArg = PyTuple_New(4);
		PyTuple_SetItem(pArg, 0, pImage);
		PyTuple_SetItem(pArg, 1, pStrPTS);
		PyTuple_SetItem(pArg, 2, pRTS);
		PyTuple_SetItem(pArg, 3, pEvents);

		CPyObject pWrap = PyTuple_New(1);
		PyTuple_SetItem(pWrap, 0, pArg);

		PyObject* pValue = PyObject_CallObject(pClass, pWrap);
		if (pValue) {
			if (pValue != Py_None) {
				PyObject* pImgRet = NULL;
				PyObject* pPtsRet = NULL;
				PyObject* pRecRet = NULL;
				if (PyTuple_Check(pValue)) {
					Py_ssize_t size = PyTuple_GET_SIZE(pValue);
					if (size > 0) pImgRet = PyTuple_GetItem(pValue, 0);
					if (size == 2) {
						PyObject* tmp = PyTuple_GetItem(pValue, 1);
						if (PyBool_Check(tmp))
							pRecRet = tmp;
						else
							pPtsRet = tmp;
					}
					if (size == 3) {
						pPtsRet = PyTuple_GetItem(pValue, 1);
						pRecRet = PyTuple_GetItem(pValue, 2);
					}
				}
				else if (PyArray_Check(pValue)) {
					pImgRet = pValue;
				}
				else {
					if (PyBool_Check(pValue))
						pRecRet = pValue;
					else
						pPtsRet = pValue;
				}

				if (pImgRet) {
					if (PyArray_Check(pImgRet)) {
						int ndims = PyArray_NDIM((const PyArrayObject*)pImgRet);
						const npy_intp* np_sizes = PyArray_DIMS((PyArrayObject*)pImgRet);
						const npy_intp* np_strides = PyArray_STRIDES((PyArrayObject*)pImgRet);

						int height = np_sizes[0];
						int width = np_sizes[1];
						int depth = np_sizes[2];
						int stride = np_strides[0];

						uint8_t* np_buf = (uint8_t*)PyArray_BYTES((PyArrayObject*)pImgRet);

						if (f.m_frame->width != width || f.m_frame->height != height) {
							AVPixelFormat pix_fmt = (AVPixelFormat)f.m_frame->format;
							int64_t pts = f.m_frame->pts;
							av_frame_free(&f.m_frame);
							f.m_frame = av_frame_alloc();
							f.m_frame->width = width;
							f.m_frame->height = height;
							f.m_frame->format = pix_fmt;
							f.m_frame->pts = pts;
							av_frame_get_buffer(f.m_frame, 0);
							av_frame_make_writable(f.m_frame);
						}

						for (int y = 0; y < height; y++)
							memcpy(f.m_frame->data[0] + y * f.m_frame->linesize[0], np_buf + y * stride, width * depth);

					}
				}

				if (pPtsRet) {
					if (pPtsRet != Py_None) {
						PyObject* repr = PyObject_Repr(pPtsRet);
						PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "strict");
						char* result = PyBytes_AsString(str);
						std::size_t pos = 0;
						std::string s(result);
						s.erase(0, 1);
						s.erase(s.length()-1, s.length());
						try {
							int64_t pts = std::stoll(s, &pos);
							f.m_frame->pts = pts;
						}
						catch (const std::exception& e) {
							std::cout << "Error interpreting python pts return value: (" << result << ") " << e.what() << std::endl;
						}
					}
				}

				if (pRecRet) {
					if (pRecRet != Py_None) {
						if (pRecRet == Py_True)
							result = true;
					}
				}
			}

			Py_DECREF(pValue);
			pValue = NULL;

			if (pData) Py_DECREF(pData);
			pData = NULL;
		}
		else {
			std::cout << "Error: pyrunner returned null pvalue" << std::endl;
		}
	}

	return result;
}

void PyRunner::test(const std::string& arg1, const std::string& arg2) 
{
    std::cout << "PyRunner::test: " << arg1 << " " << arg2 << std::endl;
}

}

void test_function() 
{
    std::cout << "test function" << std::endl;
}

Q_DECLARE_METATYPE(std::string);

int main(int argc, char *argv[])
{
    Py_Initialize();
    QApplication a(argc, argv);
    qRegisterMetaType<std::string>();
    MainWindow w;
    //std::function<void(void)> tf = test_function;

    using namespace std::placeholders;
    avio::PyRunner runner;
    std::function<void(const std::string&, const std::string&)> tf = std::bind(&avio::PyRunner::test, &runner, _1, _2);
    std::function<int(const std::string&, const std::string&, const std::string&, const std::string&)> initPy 
        = std::bind(&avio::PyRunner::initialize, &runner, _1, _2, _3, _4);
    w.fnct_ptr = tf;
    w.initPy = initPy;
    w.show();
    return a.exec();
}
