import logging as log
import sys
import argparse
import os
import venv
import urllib.request
import zipfile

def initialize_repo_and_env(llava_video=False, new_vlm=False):
    # getting the pip and python from the venv
    VENV_DIR = "openvino-env"
    if llava_video:
        VENV_DIR = "llava-video-env"
    if new_vlm:
        VENV_DIR = "new-ov-env"
    pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
    python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")

    setup_requirements = False
    # If user not in venv, creates the venv and activates
    if sys.prefix == sys.base_prefix:
        print("Not in a venv, so will create and activate.")

        VENV_DIR = "openvino-env"
        if llava_video:
            VENV_DIR = "llava-video-env"
        if new_vlm:
            VENV_DIR = "new-ov-env"
        # Creating the venv if it does not exist
        if not os.path.exists(VENV_DIR):
            venv.create(VENV_DIR, with_pip=True)
            setup_requirements = True


        pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
        python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")


        
    
    print("Activated venv")
    print(pip_exe)

    entries = os.listdir(os.getcwd())

    if new_vlm:
        if "openvino.genai_updated" in entries:
            print("Already found openvino.genai_updated, Assuming repo has been setup properly...")
            os.chdir("openvino.genai_updated")
            os.chdir("tools")
            os.chdir("llm_bench")
        else:
            os.system("git clone https://github.com/abheekga/openvino.genai_updated.git")
            os.chdir("openvino.genai_updated")
            os.chdir("tools")
            os.chdir("llm_bench")
            
        if setup_requirements:
            print("Installing the requirements from the llm_bench")
            cwd = os.getcwd()
            requirements_file = os.path.join(cwd, "requirements.txt")
            if llava_video:
                requirements_file = os.path.join(cwd, "llava_video_requirements.txt")

            if os.path.exists(requirements_file):
                print("Found req file")
            else:
                print(f"Files here: {os.listdir(cwd)}")

            os.system(f"\"{python_exe}\" -m pip install --upgrade pip")
            os.system(f"\"{pip_exe}\" install -r {requirements_file}")
            os.system(f"\"{pip_exe}\" install av")
            print("Requirements were installed.")
        
        print("Inside the repo and entered llm_bench, ready for inference")
        return python_exe

    # If openvino.genai repo already exists, navigate into correct directory
    if "openvino.genai" in entries:
        print("Already found openvino.genai, Assuming repo has been setup properly...")
        os.chdir("openvino.genai")
        os.chdir("tools")
        os.chdir("llm_bench")
    # Otherwise, clones the correct repo and then installs requirements
    else:
        os.system("git clone https://github.com/abheekga/openvino.genai.git")
        os.chdir("openvino.genai")
        os.chdir("tools")
        os.chdir("llm_bench")
        
    if setup_requirements:
        print("Installing the requirements from the llm_bench")
        cwd = os.getcwd()
        requirements_file = os.path.join(cwd, "openvino_requirements.txt")
        if llava_video:
            requirements_file = os.path.join(cwd, "llava_video_requirements.txt")

        if os.path.exists(requirements_file):
            print("Found req file")
        else:
            print(f"Files here: {os.listdir(cwd)}")

        os.system(f"\"{python_exe}\" -m pip install --upgrade pip")
        os.system(f"\"{pip_exe}\" install -r {requirements_file}")
        print("Requirements were installed.")


    print("Inside the repo and entered llm_bench, ready for inference")

    return python_exe

def initialize_repo_and_env_minicpm_o():
    # getting the pip and python from the venv
    VENV_DIR = "minicpm-o-env"
    pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
    python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")
    setup_requirements = False
    # If user not in venv, creates the venv and activates
    if sys.prefix == sys.base_prefix:
        print("Not in a venv, so will create and activate.")

        VENV_DIR = "minicpm-o-env"
        # Creating the venv if it does not exist
        if not os.path.exists(VENV_DIR):
            venv.create(VENV_DIR, with_pip=True)
            setup_requirements = True


        pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
        python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")

        
    
    print("Activated venv")
    print(pip_exe)

    entries = os.listdir(os.getcwd())

    # If openvino.genai repo already exists, navigate into correct directory
    if "openvino.genai" in entries:
        print("Already found openvino.genai, Assuming repo has been setup properly...")
        os.chdir("openvino.genai")
        os.chdir("tools")
        os.chdir("llm_bench")
        os.chdir("minicpm-o")
    # Otherwise, clones the correct repo and then installs requirements
    else:
        os.system("git clone https://github.com/abheekga/openvino.genai.git")
        os.chdir("openvino.genai")
        os.chdir("tools")
        os.chdir("llm_bench")
        os.chdir("minicpm-o")
    
    if setup_requirements:
        print("Installing the requirements from the minicpm-o")
        cwd = os.getcwd()
        requirements_file = os.path.join(cwd, "requirements.txt")

        if os.path.exists(requirements_file):
            print("Found req file")
        else:
            print(f"Files here: {os.listdir(cwd)}")

        os.system(f"\"{python_exe}\" -m pip install --upgrade pip")
        os.system(f"\"{pip_exe}\" install -r {requirements_file}")
        print("Requirements were installed.")


    print("Inside the repo and entered llm_bench, ready for inference")

    return python_exe

def initialize_repo_and_env_llama():
    # getting the pip and python from the venv
    VENV_DIR = "llama-env"
    pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
    python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")
    setup_requirements = False
    # If user not in venv, creates the venv and activates
    if sys.prefix == sys.base_prefix:
        print("Not in a venv, so will create and activate.")

        VENV_DIR = "llama-env"
        # Creating the venv if it does not exist
        if not os.path.exists(VENV_DIR):
            venv.create(VENV_DIR, with_pip=True)
            setup_requirements = True


        pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
        python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")

        
    
    print("Activated venv")
    print(pip_exe)

    entries = os.listdir(os.getcwd())

    # If openvino.genai repo already exists, navigate into correct directory
    if "openvino.genai" in entries:
        print("Already found openvino.genai, Assuming repo has been setup properly...")
        os.chdir("openvino.genai")
        os.chdir("tools")
        os.chdir("llm_bench")
    # Otherwise, clones the correct repo and then installs requirements
    else:
        os.system("git clone https://github.com/abheekga/openvino.genai.git")
        os.chdir("openvino.genai")
        os.chdir("tools")
        os.chdir("llm_bench")
    
    if setup_requirements:
        print("Installing the requirements for llm_bench")
        cwd = os.getcwd()
        requirements_file = os.path.join(cwd, "requirements.txt")

        if os.path.exists(requirements_file):
            print("Found req file")
        else:
            print(f"Files here: {os.listdir(cwd)}")

        os.system(f"\"{python_exe}\" -m pip install --upgrade pip")
        os.system(f"\"{pip_exe}\" install -r {requirements_file}")
        print("Requirements were installed.")

    os.chdir("llama")
    llamaDir = os.getcwd()
    entries = os.listdir(os.getcwd())
    if "llama.cpp" not in entries:
        os.system("git clone https://github.com/ggml-org/llama.cpp.git")
    
    if "llama-vulkan" not in entries:
        urllib.request.urlretrieve("https://github.com/ggml-org/llama.cpp/releases/download/b6567/llama-b6567-bin-win-vulkan-x64.zip", "llama-vulkan.zip")
        with zipfile.ZipFile("llama-vulkan.zip", 'r') as zip_ref:
            zip_ref.extractall("llama-vulkan")
        os.remove("llama-vulkan.zip")

    if setup_requirements:
        print("Installing the requirements from llama.cpp")
        os.chdir("llama.cpp")
        cwd = os.getcwd()
        requirements_file = os.path.join(cwd, "requirements.txt")

        if os.path.exists(requirements_file):
            print("Found req file")
        else:
            print(f"Files here: {os.listdir(cwd)}")

        os.system(f"\"{python_exe}\" -m pip install --upgrade pip")
        os.system(f"\"{pip_exe}\" install -r {requirements_file}")
        print("Requirements were installed.")

    # Make sure we are in the correct directory
    os.chdir(llamaDir)
    print("Inside the repo and entered llm_bench, ready for inference")

    return python_exe

def main(args):
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    if args.framework == "llama.cpp":
        python_exe = initialize_repo_and_env_llama()
        scripts_dir = os.path.dirname(python_exe)

        # Setting the correct environment path to properly use the venv packages
        os.environ["PATH"] = f"{scripts_dir};{os.environ.get('PATH','')}"

        cmd_line = f"{python_exe} llama-pipeline.py --input {args.input} --output {args.output}"
        if args.model:
            cmd_line += f" -m {args.model}"
        if args.mem:
            cmd_line += " --mem"
        if args.all:    
            cmd_line += " --all"
        ret = os.system(cmd_line)
        if ret != 0:
            sys.exit(ret)

        sys.exit(0)
    if args.category == "vlm" and args.model == "minicpm-o":
        python_exe = initialize_repo_and_env_minicpm_o()
        scripts_dir = os.path.dirname(python_exe)

        # Setting the correct environment path to properly use the venv packages
        os.environ["PATH"] = f"{scripts_dir};{os.environ.get('PATH','')}"
        
        ret = os.system(f"{python_exe} minicpm-o-omnimodal-chatbot.py")
        if ret != 0:
            sys.exit(ret)

        sys.exit(0)
    else:
        if args.category == "vlm" and args.model == "llava-video":
            python_exe = initialize_repo_and_env(llava_video=True)
        elif args.category == "vlm" and args.config == True:
            python_exe = initialize_repo_and_env(new_vlm=True)
        else:
            python_exe = initialize_repo_and_env()
        scripts_dir = os.path.dirname(python_exe)

        # Setting the correct environment path to properly use the venv packages
        os.environ["PATH"] = f"{scripts_dir};{os.environ.get('PATH','')}"

        # Depending on the category of user request, runs the proper command
        if args.category == "llm":
            if args.mem:
                ret = os.system(f"{python_exe} llm-pipeline-ov.py -m {args.model} --input {args.input} --output {args.output} --mem")
            else:
                ret = os.system(f"{python_exe} llm-pipeline-ov.py -m {args.model} --input {args.input} --output {args.output}")
            if ret != 0:
                sys.exit(ret)
        elif args.category == "vlm":
            if args.mem:
                ret = os.system(f"{python_exe} vlm-pipeline-ov.py -m {args.model} --input {args.input} --output {args.output} --height {args.height} --width {args.width} --mem")
            else:
                ret = os.system(f"{python_exe} vlm-pipeline-ov.py -m {args.model} --input {args.input} --output {args.output} --height {args.height} --width {args.width}")
            if ret != 0:
                sys.exit(ret)

        sys.exit(0)
    


   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m")
    parser.add_argument("--category", "-c", required=True)
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    parser.add_argument("--height", default=512)
    parser.add_argument("--width", default=512)
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--mem", action="store_true")
    parser.add_argument("--framework", default="genai", choices=["genai", "llama.cpp"], help="Benchmarking framework to use, default is genai")
    parser.add_argument("--all", default=False, action="store_true", help="Run all models sequentially")
    args=parser.parse_args()
    main(args)