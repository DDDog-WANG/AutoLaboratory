from mujoco.glfw import glfw
import mujoco as mj

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def create_scroll_callback(model, scene, cam):
    def callback_scroll(window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                        yoffset, scene, cam)
    return callback_scroll


def create_keyboard_callback(model, data):
    def callback_keyboard(window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
    return callback_keyboard


def create_mouseButton_callback():
    def callback_mouse_button(window, button, act, mods):
        global button_left
        global button_middle
        global button_right

        button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        glfw.get_cursor_pos(window)
    return callback_mouse_button


def create_mouse_move_callback(model, scene, cam):
    def callback_mouse_move(window, xpos, ypos):
        global lastx
        global lasty
        global button_left
        global button_middle
        global button_right

        dx = xpos - lastx
        dy = ypos - lasty
        lastx = xpos
        lasty = ypos

        # no buttons down: nothing to do
        if (not button_left) and (not button_middle) and (not button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(model, action, dx/height,
                        dy/height, scene, cam)
    return callback_mouse_move