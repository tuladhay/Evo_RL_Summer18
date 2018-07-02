Copy these environments to wherever your gym models are located.
Copy the xml files to wherever your gym model xmls are located.
for example: gym/gym/envs/mujoco    (for model envs)
for example: gym/gym/envs/mujoco/assets    (for model xmls)

Add the environments to the "__init__.py" file (/gym/gym/envs/mujoco)
Also in order for gym to "register" it as an environment, add it to the "__init__.py" in (/gym/gym/envs). Follow how other environments are written there.
