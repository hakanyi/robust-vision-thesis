#!/usr/bin/env python3
from mitsuba.core.xml import load_dict

def load_obj(obj_pth):
    m = load_dict({
            "type" : "obj",
            "filename" : obj_pth,
    })
    verts = m.vertex_positions_buffer()
    faces = m.faces_buffer()
    return verts, faces
