import mujoco
import mujoco_viewer


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path('assets/dobot_suction_cup.xml')
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data, title="Dobot magician")
    while True:
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break
    viewer.close()