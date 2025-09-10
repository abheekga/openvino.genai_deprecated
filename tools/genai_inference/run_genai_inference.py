import logging as log
import sys
import argparse
import os
import venv

def initialize_repo_and_env(llava_video=False, new_vlm=False):
    # getting the pip and python from the venv
    VENV_DIR = "openvino-env"
    if llava_video:
        VENV_DIR = "llava-video-env"
    if new_vlm:
        VENV_DIR = "new-ov-env"
    pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
    python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")

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


        pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
        python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")

        os.execv(python_exe, [python_exe] + sys.argv)
    
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

    # If user not in venv, creates the venv and activates
    if sys.prefix == sys.base_prefix:
        print("Not in a venv, so will create and activate.")

        VENV_DIR = "minicpm-o-env"
        # Creating the venv if it does not exist
        if not os.path.exists(VENV_DIR):
            venv.create(VENV_DIR, with_pip=True)


        pip_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "pip.exe")
        python_exe = os.path.join(os.getcwd(), VENV_DIR, "Scripts", "python.exe")

        os.execv(python_exe, [python_exe] + sys.argv)
    
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

def main(args):
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

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
            ret = os.system(f"{python_exe} llm-pipeline-ov.py -m {args.model} --input {args.input} --output {args.output}")
            if ret != 0:
                sys.exit(ret)
        elif args.category == "vlm":
            ret = os.system(f"{python_exe} vlm-pipeline-ov.py -m {args.model} --input {args.input} --output {args.output} --height {args.height} --width {args.width}")
            if ret != 0:
                sys.exit(ret)

        sys.exit(0)
    


   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--category", "-c", required=True)
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    parser.add_argument("--height", default=512)
    parser.add_argument("--width", default=512)
    parser.add_argument("--config", default=False)
    args=parser.parse_args()
    main(args)