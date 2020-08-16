import configparser


def read_config():
    config_ini = configparser.ConfigParser()
    config_ini.read("utils/config.ini")
    return config_ini
