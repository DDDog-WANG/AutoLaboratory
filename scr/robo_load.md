<!--If you want to import your own model like robot, gripper, etc. , you should do as follows.-->

<u>Robosuite dir in my case is [~/.local/lib/python3.8/site-packages]()</u>



# Add robot

1. [/robosuite/models/robots/manipulators/]()
   
   * Create `maholo_robot.py`
   
   * Add `Maholo` class to `__init.py__`
2. [/robosuite/models/assets/robots/]()
   
   * mkdir `maholo` folder, under this folder do:
     * Create `robot.xml`
     * mkdir `meshes` folder, then put `.stl` files under it
3. [/robosuite/robots/]()
   
   * Add `"Maholo": Bimanual` to `__init.py__`



# Add gripper

1. [/robosuite/models/grippers/]()
   
   * Create `maholo_gripper.py`
   
   * Add `MaholoGripper` class to `__init.py__`
2. [/robosuite/models/assets/grippers/]()
   
   * Create `maholo_gripper.xml`
   
   * Under `meshes` folder, mkdir `maholo_gripper` folder and add `.stl` files 



# Add mount

1. [/robosuite/models/mounts/]()

   * Create `maholo_mount.py`

   * Add `MaholoMount` class to `__init.py__`

2. [/robosuite/models/assets/mounts/]()

   * Create `maholo_mount.xml`

   * Under `meshes` folder, mkdir `maholo_mount` folder and add `.stl` files 



# Load environment

1. [/robosuite/environment/manipulation/]()

   * Create `maholo_laboratory.py`
   * Add `MaholoLaboratory` class to `__init.py__`

2. [/robosuite/models/]()

   * [arenas]()

     * Create `name_arenas.py`

     * Add `NameArenas` class to `__init.py__`

   * [objects]()
     * Add `NameObject` class `xml_objects.py`
     * Add `NameObject` class `__init.py__`

3. [/robosuite/models/asserts/]()

   * [arenas]()
     * Create `name_arenas.xml`

   * [objects]()
     * Create `name_object.xml`
     * Under `meshes` folder, Add `.stl`, `.obj` files



# Load controllers

1. [/robosuite/controllers/]()



