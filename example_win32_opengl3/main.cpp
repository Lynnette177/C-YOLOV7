#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "yolo.h"
#include <sstream>
#include <ctime>

#include <fstream>
#include <iostream>
#include<opencv.hpp>
#include <SDKDDKVer.h>
#include <stdio.h>
#include <tchar.h>
#include<math.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/GL.h>
#include <tchar.h>
#define USE_CUDA true //use opencv-cuda


using namespace std;
using namespace cv;
using namespace dnn;


// Data stored per platform window
struct WGL_WindowData { HDC hDC; };

// Data
static HGLRC            g_hRC;
static WGL_WindowData   g_MainWindow;
static int              g_Width;
static int              g_Height;

// Forward declarations of helper functions
bool CreateDeviceWGL(HWND hWnd, WGL_WindowData* data);
void CleanupDeviceWGL(HWND hWnd, WGL_WindowData* data);
void ResetDeviceWGL();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Main code


Point sp(-1, -1);
Point ep(-1, -1);
Mat temp;
Mat roi;
Rect boxglobal(1, 1,1, 1);
static void on_draw(int event, int x, int y, int flags, void* userdata)
{
    Mat image = *((Mat*)userdata);
    if (event == EVENT_LBUTTONDOWN)
    {
        sp.x = x;
        sp.y = y;
        std::cout << "start point: " << sp << std::endl;
    }

    else if (event == EVENT_MOUSEMOVE)
    {
        if (sp.x > 0 && sp.y > 0)
        {
            ep.x = x;
            ep.y = y;
            int dx = ep.x - sp.x;
            int dy = ep.y - sp.y;
            if (dx > 0 && dy > 0)
            {
                Rect box(sp.x, sp.y, dx, dy);
                temp.copyTo(image);
                rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
                imshow("Select", image);
            }
        }
    }
    else if (event == EVENT_LBUTTONUP)
    {
        ep.x = x;
        ep.y = y;
        cout << ep.x << endl;
        cout << ep.y << endl;
        if (ep.x > 1024) ep.x = 1024;
        if (ep.y > 768) ep.y = 768;//防止越界导致错误
        int dx = ep.x - sp.x;
        int dy = ep.y - sp.y;
        //dy = dx;
        if (dx > 0 && dy > 0)
        {
            Rect box(sp.x, sp.y, dx, dy);
            boxglobal = box;
            roi = temp(box);
            rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
            //更新起始点
            sp.x = -1;
            sp.y = -1;
        }

    }
}

void mouse_drawing_demo(Mat& image)
{
    namedWindow("Select", WINDOW_FREERATIO);
    setMouseCallback("Select", on_draw, (void*)(&image));
    imshow("Select", image);
    temp = image.clone();
}

int main(int, char**)
{
    ShowWindow(GetConsoleWindow(), SW_HIDE);
    // 创建应用程序窗口
    WNDCLASSEXW wc = { sizeof(wc), CS_OWNDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, L"ImGui Example", NULL };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"YOLOV7表情识别系统", WS_OVERLAPPEDWINDOW, 100, 100, 800, 600, NULL, NULL, wc.hInstance, NULL);

    // 初始化OpenGL
    if (!CreateDeviceWGL(hwnd, &g_MainWindow))
    {
        CleanupDeviceWGL(hwnd, &g_MainWindow);
        ::DestroyWindow(hwnd);
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    wglMakeCurrent(g_MainWindow.hDC, g_hRC);

    // 显示窗口
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // 设置Dear ImGui上下文
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\simhei.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;    // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplWin32_InitForOpenGL(hwnd);
    ImGui_ImplOpenGL3_Init();



    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    string model_path = "best emotion.onnx";

    Yolo test;
    Net net;
    if (test.readModel(net, model_path, USE_CUDA)) {
        cout << "成功加载onnx" << endl;
    }
    else {
        cout << "加载onnx失败" << endl;
        return -1;
    }

    //生成随机颜色
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }
    vector<Output> result;
    vector<Output> newresult;
    VideoCapture capture(0);
    capture.set(3, 1024);
    capture.set(4, 768);
    // Main loop
    bool done = false;
    while (!done)
    {
        // 处理窗口事件消息，鼠标点击等等
        MSG msg;
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();


        Mat img;

        capture.read(img);//读取视频
        
        flip(img, img, 1);//图像镜像操作
        if (img.empty())
        {
            break;
        }
        Mat rimg;
        roi = img;
        imshow("Select", img);
        rimg = img(boxglobal);
        mouse_drawing_demo(img);


        int dected = 0;
        if (test.Detect(rimg, net, result))
        {
            test.drawPred(rimg, result, color);
            dected = 1;
        }
        else
        {
            test.drawPred(rimg, result, color);
            //未检测到任何相关信息
            dected = 0;
        }

        result = newresult;

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin(u8"表情识别系统");                          // Create a window called "Hello, world!" and append into it.
            if(USE_CUDA)
            ImGui::Text(u8"使用GPU cuda核心");               // Display some text (you can use a format strings too)
            else ImGui::Text(u8"使用CPU");
            if (dected) ImGui::Text(u8"已识别到表情");
            else ImGui::Text(u8"未能识别到任何表情");
            ImGui::Spacing;
            ImGui::Text(u8"总人数：%d", test.personnum);
            ImGui::Spacing;
            int sumemotion = 0;
            for (int z = 0; z <= 7; z++) {
                sumemotion += test.emotion[z];
            }
            ImGui::Text(u8"生气：%d", test.emotion[0]);
            if(sumemotion!=0)
            ImGui::Text(u8"生气占比：%.1f%%", 100 * float(test.emotion[0])/ float(sumemotion));
            else ImGui::Text(u8"生气占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"不屑：%d", test.emotion[1]);
            if (sumemotion != 0)
                ImGui::Text(u8"不屑占比：%.1f%%", 100 * float(test.emotion[1]) / float(sumemotion));
            else ImGui::Text(u8"不屑占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"厌恶：%d", test.emotion[2]);
            if (sumemotion != 0)
                ImGui::Text(u8"厌恶占比：%.1f%%", 100 * float(test.emotion[2]) / float(sumemotion));
            else ImGui::Text(u8"厌恶占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"恐惧：%d", test.emotion[3]);
            if (sumemotion != 0)
                ImGui::Text(u8"恐惧占比：%.1f%%", 100 * float(test.emotion[3]) / float(sumemotion));
            else ImGui::Text(u8"恐惧占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"高兴：%d", test.emotion[4]);
            if (sumemotion != 0)
                ImGui::Text(u8"高兴占比：%.1f%%",100* float(test.emotion[4]) / float(sumemotion));
            else ImGui::Text(u8"高兴占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"中性：%d", test.emotion[5]);
            if (sumemotion != 0)
                ImGui::Text(u8"中性占比：%.1f%%", 100 * float(test.emotion[5]) / float(sumemotion));
            else ImGui::Text(u8"中性占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"难过：%d", test.emotion[6]);
            if (sumemotion != 0)
                ImGui::Text(u8"难过占比：%.1f%%", 100 * float(test.emotion[6]) / float(sumemotion));
            else ImGui::Text(u8"难过占比：%f%%", 0);
            ImGui::Spacing;
            ImGui::Text(u8"惊讶：%d", test.emotion[7]);
            if (sumemotion != 0)
                ImGui::Text(u8"惊讶占比：%.1f%%", 100 * float(test.emotion[7]) / float(sumemotion));
            else ImGui::Text(u8"惊讶占比：%f%%", 0);

            if (ImGui::Button(u8"重置")) {
                for (int z = 0; z <= 7; z++) {
                    test.emotion[z]=0;
                }
            }
            if (ImGui::Button(u8"保存当前数据")) {
                //cout<<(timeStr.str());
                time_t t = time(nullptr);
                struct tm* now = localtime(&t);

                std::stringstream timeStr;

                // 以下依次把年月日的数据加入到字符串中
                timeStr << "Results\\";
                timeStr << now->tm_year + 1900 << "-";
                timeStr << now->tm_mon + 1 << "-";
                timeStr << now->tm_mday << "-";
                timeStr << now->tm_hour << "-";
                timeStr << now->tm_min << "-";
                timeStr << now->tm_sec;
                timeStr << ".txt";
                string filename = timeStr.str();
                const char* p = filename.c_str();
               // cout << p;
                ofstream out(p);
                if (out.is_open())
                {
                    out << "总人数：" << test.personnum << endl<<endl;
                    out << "生气：" << test.emotion[0] << endl;
                    out << "不屑：" << test.emotion[1] << endl;
                    out << "厌恶：" << test.emotion[2] << endl;
                    out << "恐惧：" << test.emotion[3] << endl;
                    out << "高兴：" << test.emotion[4] << endl;
                    out << "中性：" << test.emotion[5] << endl;
                    out << "难过：" << test.emotion[6] << endl;
                    out << "惊讶：" << test.emotion[7] << endl;
                    out << "\n";
                    out << "生气占比：" << 100 * float(test.emotion[0]) / float(sumemotion) <<"%" << endl;
                    out << "不屑占比：" << 100 * float(test.emotion[1]) / float(sumemotion) << "%" << endl;
                    out << "厌恶占比：" << 100 * float(test.emotion[2]) / float(sumemotion) << "%" << endl;
                    out << "恐惧占比：" << 100 * float(test.emotion[3]) / float(sumemotion) << "%" << endl;
                    out << "高兴占比：" << 100 * float(test.emotion[4]) / float(sumemotion) << "%" << endl;
                    out << "中性占比：" << 100 * float(test.emotion[5]) / float(sumemotion) << "%" << endl;
                    out << "难过占比：" << 100 * float(test.emotion[6]) / float(sumemotion) << "%" << endl;
                    out << "惊讶占比：" << 100 * float(test.emotion[7]) / float(sumemotion) << "%" << endl;
                    out.close();
                }
                
            }
           
            ImGui::Checkbox(u8"显示侦测窗口", &test.window);

            ImGui::Text(u8"应用程序的帧率是： (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        

        // Rendering
        ImGui::Render();
        glViewport(0, 0, g_Width, g_Height);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Present
        ::SwapBuffers(g_MainWindow.hDC);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceWGL(hwnd, &g_MainWindow);
    wglDeleteContext(g_hRC);
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    /*这段代码主要用于清理和释放在初始化过程中创建的资源和上下文。具体的操作包括：
ImGui_ImplOpenGL3_Shutdown();：关闭并清理Dear ImGui的OpenGL后端。
ImGui_ImplWin32_Shutdown();：关闭并清理Dear ImGui的Win32后端。
ImGui::DestroyContext();：销毁Dear ImGui的上下文。
CleanupDeviceWGL(hwnd, &g_MainWindow);：清理OpenGL设备。
wglDeleteContext(g_hRC);：删除OpenGL渲染上下文。
::DestroyWindow(hwnd);：销毁应用程序窗口。
::UnregisterClassW(wc.lpszClassName, wc.hInstance);：取消注册窗口类。
最后，函数返回0表示成功执行完程序，没有错误。确保在程序退出时释放所有使用的资源，避免内存泄漏和其他问题。*/
    return 0;
}

bool CreateDeviceWGL(HWND hWnd, WGL_WindowData* data)
{
    HDC hDc = ::GetDC(hWnd);
    PIXELFORMATDESCRIPTOR pfd = { 0 };
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;

    const int pf = ::ChoosePixelFormat(hDc, &pfd);
    if (pf == 0)
        return false;
    if (::SetPixelFormat(hDc, pf, &pfd) == FALSE)
        return false;
    ::ReleaseDC(hWnd, hDc);

    data->hDC = ::GetDC(hWnd);
    if (!g_hRC)
        g_hRC = wglCreateContext(data->hDC);
    return true;
}

void CleanupDeviceWGL(HWND hWnd, WGL_WindowData* data)
{
    wglMakeCurrent(NULL, NULL);
    ::ReleaseDC(hWnd, data->hDC);
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED)
        {
            g_Width = LOWORD(lParam);
            g_Height = HIWORD(lParam);
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
