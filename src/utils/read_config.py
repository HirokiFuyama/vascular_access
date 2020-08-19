import configparser


def read_config():
    config_ini = configparser.ConfigParser()
    config_ini.read("/Users/hiroki/github/vascular_access/src/utils/config.ini")
    return config_ini
