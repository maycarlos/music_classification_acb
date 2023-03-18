from dotenv import dotenv_values, find_dotenv


# Env file loading
def load_env():
    env_file = find_dotenv()
    config = dotenv_values(env_file)
    return config


ENV_CONFIG = load_env()