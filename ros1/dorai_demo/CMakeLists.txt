cmake_minimum_required(VERSION 3.0.2)
project(dorai_demo)

find_package(catkin REQUIRED)
catkin_package()

set(DORAI_SCRIPTS
  person_node.py
  bridge_node.py
  behaviour_node.py
)

# Install space: the normal thing, a copy.
foreach(script ${DORAI_SCRIPTS})
  list(APPEND DORAI_SCRIPT_PATHS scripts/${script})
endforeach()
catkin_install_python(PROGRAMS ${DORAI_SCRIPT_PATHS}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Devel space: SYMLINK to the source instead of copying.
#
# catkin_install_python copies the scripts into the devel space at configure
# time and then does not refresh them when the source changes, not even with
# --force-cmake. The result is that you edit a node, rerun it, and silently get
# the previous version. That was observed here and cost real debugging time, and
# it is precisely the failure you do not want while tuning gains in front of an
# audience. Symlinking means rosrun and roslaunch always execute the file you
# just edited; only a genuinely new script needs a rebuild.
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})
foreach(script ${DORAI_SCRIPTS})
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${script}
                  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/${script})
endforeach()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
