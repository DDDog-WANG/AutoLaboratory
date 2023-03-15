from mujoco.glfw import glfw
import mujoco as mj
from mouseKey import *

def make_glfw_window(model, data, cam, opt):
    # Init GLFW, create window, make OpenGL context current, request v-sync
    glfw.init()
    window = glfw.create_window(1200, 900, "Demo", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # initialize visualization data structures
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)



    # install GLFW mouse and keyboard callbacks
    glfw.set_key_callback(window,          create_keyboard_callback(model, data))
    glfw.set_cursor_pos_callback(window,   create_mouse_move_callback(model, scene, cam))       #
    glfw.set_mouse_button_callback(window, create_mouseButton_callback())                       # 鼠标按键检测回调函数
    glfw.set_scroll_callback(window,       create_scroll_callback(model, scene, cam))           # 中键滚动回调函数
    return window, scene, context


def window_update(model, data, opt, cam, scene, window, context):
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()