import configparser


def read_config():
    config_ini = configparser.ConfigParser()
    config_ini.read("/vascular_access/src/utils/config.ini")
    return config_ini
