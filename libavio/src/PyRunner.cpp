/********************************************************************
* libavio/src/PyRunner.cpp
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

#include "PyRunner.h"

namespace avio
{

PyRunner::PyRunner()
{
    Py_Initialize();
}

PyRunner::~PyRunner()
{
	if (pClass) Py_DECREF(pClass);
    Py_Finalize();
}

int PyRunner::initialize(const std::string& dir, const std::string& file, const std::string& py_class, const std::string& arg)
{
    import_array1(1);
    try {
		PyObject* sysPath = PySys_GetObject("path");
		PyObject* dirName = PyUnicode_FromString(dir.c_str());
		PyList_Append(sysPath, dirName);

		PyObject* pName = PyUnicode_FromString(file.c_str());		     if (!pName)   throw Exception("pName");
		PyObject* pModule = PyImport_Import(pName);                      if (!pModule) throw Exception("pModule");
		PyObject* pDict = PyModule_GetDict(pModule);                     if (!pDict)   throw Exception("pDict");
		PyObject* pItem = PyDict_GetItemString(pDict, py_class.c_str()); if (!pItem)   throw Exception("pItem");

		PyObject* pWrap = nullptr;
		PyObject* pArg = nullptr;

		if (arg.length() > 0) {
			pArg = Py_BuildValue("(s)", arg.c_str());
			pWrap = PyTuple_New(1);
			PyTuple_SetItem(pWrap, 0, pArg);
		}

		pClass = PyObject_CallObject(pItem, pWrap);                      if (!pClass) throw Exception("pClass");

		if (sysPath) Py_DECREF(sysPath);
		if (dirName) Py_DECREF(dirName);
		if (pName)   Py_DECREF(pName);
		if (pModule) Py_DECREF(pModule);
		if (pDict)   Py_DECREF(pDict);
		if (pItem)   Py_DECREF(pItem);
		if (pArg)    Py_DECREF(pArg);
		if (pWrap)   Py_DECREF(pWrap);
	}
	catch (const Exception& e) {
		std::cout << "PyRunner constuctor exception: " << e.what() << std::endl;
	}

    return 0;
}

PyObject* PyRunner::getImage(Frame& f)
{
	PyObject* result;
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
		if (mat_buf) delete[] mat_buf;
		mat_buf = new uint8_t[linesize * height];

		for (int y = 0; y < height; y++)
			memcpy(mat_buf + y * linesize, f.m_frame->data[0] + y * f.m_frame->linesize[0], linesize);

		npy_intp dimensions[3] = { f.m_frame->height, f.m_frame->width, depth };
		pData = PyArray_SimpleNewFromData(3, dimensions, NPY_UINT8, mat_buf);
		if (!pData) throw Exception("pData");
		result = Py_BuildValue("(O)", pData);
        if (!result) throw Exception("Py_BuildValue null return");
	}
	catch (const Exception& e) {
		ex.msg(e.what(), MsgPriority::CRITICAL, "PyRunner::getImage exception: ");
	}
	return result;
}

bool PyRunner::run(Frame& f)
{
	bool result = false;

	if (f.isValid()) {
		PyObject* pImage = getImage(f);
		PyObject* pRTS = Py_BuildValue("(i)", f.m_rts);

		// pts is long long so use stream writer to convert
		std::stringstream str;
		str << f.m_frame->pts;
		PyObject* pStrPTS = Py_BuildValue("(s)", str.str().c_str());

		PyObject* pArg = PyTuple_New(3);
		PyTuple_SetItem(pArg, 0, pImage);
		PyTuple_SetItem(pArg, 1, pStrPTS);
		PyTuple_SetItem(pArg, 2, pRTS);

		PyObject* pWrap = PyTuple_New(1);
		PyTuple_SetItem(pWrap, 0, pArg);

		PyObject* pValue = PyObject_CallObject(pClass, pWrap);
		if (pValue) {
			if (pValue != Py_None) {
				PyObject* pImgRet = nullptr;
				PyObject* pPtsRet = nullptr;
				PyObject* pRecRet = nullptr;
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

			if (pValue) Py_DECREF(pValue);
			pValue = nullptr;

			if (pData) Py_DECREF(pData);
			pData = nullptr;

			if (pImage)  Py_DECREF(pImage);
			if (pRTS)    Py_DECREF(pRTS);
			if (pStrPTS) Py_DECREF(pStrPTS);
			if (pWrap)   Py_DECREF(pWrap);
			if (pArg)    Py_DECREF(pArg);

		}
		else {
			std::cout << "Error: pyrunner returned null pvalue" << std::endl;
		}
	}

	return result;
}

}
