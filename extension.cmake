# Inkling-2-Mini Gufo extension.
# Host tier: probe and registration (no GPU context). HIP tier: the engine kernels. Text tier: runner, continuation
# adapter, provider registration.
gufo_extension_target(
  NAME gufo_inkling
  SOURCES
    gufo/probe.cpp
    gufo/extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)
if(TARGET gufo_inkling)
  target_include_directories(gufo_inkling PUBLIC "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>")
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_inkling PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

gufo_extension_target(
  NAME gufo_inkling_hip
  SOURCES
    host/tokenizer.cpp
    host/chat_template.cpp
  LINK_LIBRARIES
    gufo_inkling
    gufo_ext_runtime
    ICU::uc
    ICU::i18n
    hip::host
    Threads::Threads
)
if(TARGET gufo_inkling_hip)
  gufo_extension_hip_sources(gufo_inkling_hip
    SOURCES
      engine/engine_api.hip.cpp
  )
  target_compile_definitions(gufo_inkling_hip PRIVATE ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_inkling_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
  target_include_directories(gufo_inkling_hip PUBLIC "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>")
endif()

gufo_extension_target(
  NAME gufo_inkling_text
  SOURCES
    gufo/continuation_adapter.cpp
    gufo/runner.cpp
    gufo/provider.cpp
  LINK_LIBRARIES
    gufo_inkling_hip
    gufo_inkling
    gufo_ext_cache
    gufo_ext_runtime
    hip::host
    Threads::Threads
)
if(TARGET gufo_inkling_text)
  target_include_directories(gufo_inkling_text PUBLIC "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>")
endif()
