import proplot as pro

def setup_rc() -> None:
    pro.rc["figure.dpi"] = 300
    pro.rc["title.size"] = 8
    pro.rc["font.size"] = 7
    pro.rc["label.size"] = 7
    pro.rc["legend.fontsize"] = 6
    pro.rc["cycle"] = "ggplot"
    pro.rc["image.origin"] = "lower"
    pro.rc["image.cmap"] = "bone"
