import os
import torch
import mitsuba
# Set the desired mitsuba variant
mitsuba.set_variant('gpu_autodiff_rgb')

from robust_vision.utils.graphics import GaussianSmoothing
import robust_vision.utils.enoki as uek
import enoki as ek
from mitsuba.core import ScalarTransform4f, Transform4f, Vector3f, UInt32, Float32
from mitsuba.core.xml import load_dict
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render_torch
from skimage.filters import threshold_otsu


def bbox_mask(alpha):
    """ alpha is torch bool with shape (1,w,h), returns bbox coordinates """
    alpha = alpha.squeeze(0)
    rows = torch.any(alpha, axis=1)
    cols = torch.any(alpha, axis=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

class Renderer:
    def __init__(self, cam_azimuth=0, cam_elevation=0, cam_distance=3.5,
                 cam_translation=(0,0,0), object_color=(1.,1.,1.),
                 lamp_radiance=10, lamp_origin=(2,0,0), lamp_up=(0,1,0),
                 wall_length=1.6, wall_distance=1.6, wall_color=(1.,1.,1.),
                 res=224, sample_count=32, file_pth="meshes/grid_nogradients.obj",
                 to_unitsphere=False, center_bbox=False, face_normals=False,
                 max_depth=3, silhouette=False,
                 mooney_thresh_method="mean_mask"):

        file_type = str.split(os.path.split(file_pth)[-1], ".")[-1]
        self.requires_grad = False

        cam_translation = ScalarTransform4f.translate(v=cam_translation)
        cam_rotation = (ScalarTransform4f.rotate(axis=(0,1,0), angle=cam_azimuth) *
                        ScalarTransform4f.rotate(axis=(0,0,1), angle=cam_elevation))

        self.cam_toworld = (cam_translation * cam_rotation *
                            ScalarTransform4f.look_at(origin=(cam_distance,0,0),
                                                      target=(0, 0, 0),
                                                      up=(0, 1, 0)))
        if silhouette:
            lamp = {
                "type" : "constant"
            }
            object_color = (0., 0., 0.)
        else:
            self.lamp_toworld = (
                ScalarTransform4f.look_at(origin=lamp_origin,
                                          target=(0, 0, 0),
                                          up=lamp_up) *
                ScalarTransform4f.scale(v=0.5))
            lamp = {
                "id" : "Lamp",
                "type" : "rectangle",
                "to_world" : self.lamp_toworld,
                "emitter" : {
                    "type" : "smootharea",
                    "radiance" : float(lamp_radiance),
                },
            }
            object_color = object_color

        # wall_bsdf = {
        #     "type" : "diffuse",
        #     "reflectance" : {
        #         "type" : "rgb",
        #         "value" : wall_color,
        #     }
        # }

        self.scene = load_dict({
            "type" : "scene",
            "integrator" : {
                "type" : "aov",
                "aovs" : "d:depth",
                "integrator" : {
                    "type" : "pathreparam",
                    "max_depth" : max_depth,
                },
            },
            "sensor" : {
                "type" : "perspective",
                "to_world" : self.cam_toworld,
                # emulating pyrender IntrinsicsCamera
                "near_clip" : 0.05,
                "far_clip" : 100.,
                "film" : {
                    "type" : "hdrfilm",
                    "width" : res,
                    "height" : res,
                },
                "sampler" : {
                    "type" : "independent",
                    "sample_count" : sample_count,
                }
            },
            "lamp" : lamp,
            "shape" : {
                "id" : "Object",
                "type" : file_type,
                "filename" : file_pth,
                "face_normals" : face_normals,
                "bsdf" : {
                    "type" : "twosided",
                    "bsdf" : {
                        "type" : "diffuse",
                        "reflectance" : {
                            "type" : "rgb",
                            "value" : object_color,
                        },
                    },
                },
            },
            # "shape1" : {
            #     "id" : "right_wall",
            #     "type" : "rectangle",
            #     "to_world": (ScalarTransform4f.look_at(origin=(0,0,-wall_distance),
            #                                            target=(0,0,0),
            #                                            up=(0,1,0)) *
            #                  ScalarTransform4f.scale(v=wall_length)),
            #     # "emitter" : {
            #     #     "type" : "smootharea",
            #     #     "radiance" : wall_radiance,
            #     # },
            #     "bsdf" : wall_bsdf,
            # },
            # "shape2" : {
            #     "id" : "left_wall",
            #     "type" : "rectangle",
            #     "to_world": (ScalarTransform4f.look_at(origin=(0,0,wall_distance),
            #                                            target=(0,0,0),
            #                                            up=(0,1,0)) *
            #                  ScalarTransform4f.scale(v=wall_length)),
            #     "bsdf" : wall_bsdf,
            # },
            # "shape3" : {
            #     "id" : "bottom_wall",
            #     "type" : "rectangle",
            #     "to_world": (ScalarTransform4f.look_at(origin=(0,-wall_distance,0),
            #                                            target=(0,0,0),
            #                                            up=(0,1,0)) *
            #                  ScalarTransform4f.scale(v=wall_length)),
            #     "bsdf" : wall_bsdf,
            # },
            # # "shape4" : {
            # #     "id" : "top_wall",
            # #     "type" : "rectangle",
            # #     "to_world": (cam_rotation *
            # #                  ScalarTransform4f.translate(v=[0, wall_distance, 0]) *
            # #                  ScalarTransform4f.rotate(axis=[1,0,0], angle=90) *
            # #                  ScalarTransform4f.scale(v=wall_length)),
            # #     "bsdf" : wall_bsdf,
            # # },
            # "shape5" : {
            #     "id" : "front_wall",
            #     "type" : "rectangle",
            #     "to_world": (cam_rotation *
            #                  ScalarTransform4f.translate(v=[cam_distance+0.5, 0, 0]) *
            #                  ScalarTransform4f.rotate(axis=[0,1,0], angle=270) *
            #                  ScalarTransform4f.scale(v=2)),
            #     "bsdf" : wall_bsdf,
            # },
        })
        self.params = traverse(self.scene)
        self.preprocess_mesh(to_unitsphere=to_unitsphere,
                             center_bbox=center_bbox)
        self.faces_size = ek.slices(self.params["Object.faces_buf"])
        self.params_optim = None
        self.params_optim_torch = None
        self.gray = None
        self.alpha = None
        self.mooney_thresh_method = mooney_thresh_method
        self.smoother = GaussianSmoothing(1, 10, 2).cuda()

    def rotate_lamp(self, azimuth, elevation):
        """ rotates lamp around x-axis (elevation) and y-axis (azimuth)
        in lamp coordinates """
        xaxis = self.lamp_toworld.transform_vector([1,0,0])
        yaxis = self.lamp_toworld.transform_vector([0,1,0])
        self.params['Lamp.to_world'] = (
            ScalarTransform4f.rotate(axis=yaxis, angle=azimuth) *
            ScalarTransform4f.rotate(axis=xaxis, angle=elevation) *
            self.lamp_toworld
            )
        self.params.set_dirty('Lamp.to_world')
        self.params.update()

    def preprocess_mesh(self,
                        center_bbox=False,
                        to_unitsphere=False,
                        buf=1.03):
        """ Center and normalize the mesh """
        if to_unitsphere or center_bbox:
            verts = uek.ravel(self.params['Object.vertex_positions_buf'])
            transl = (uek.to_each_col(verts, ek.hmax) + \
                uek.to_each_col(verts, ek.hmin)) / 2.
            verts -= transl
        if to_unitsphere:
            max_distance = ek.hmax(ek.norm(verts)) * buf
            verts /= max_distance
        if to_unitsphere or center_bbox:
            uek.unravel(verts, self.params['Object.vertex_positions_buf'])
            self.params.set_dirty('Object.vertex_positions_buf')
            self.params.update()

    def release_memory(self):
        ek.cuda_malloc_trim()

    def get_vertex_grad(self):
        return ek.gradient(self.params_optim).torch()

    def rotate_mesh(self, rot_angles):
        """ rotates the mesh in the scene by rot_angles """
        rotate = (Transform4f.rotate(axis=[1, 0, 0], angle=rot_angles[0]) *
                  Transform4f.rotate(axis=[0, 1, 0], angle=rot_angles[1]) *
                  Transform4f.rotate(axis=[0, 0, 1], angle=rot_angles[2]))
        old_buf = uek.ravel(self.params['Object.vertex_positions_buf'])
        new_buf = rotate.transform_point(old_buf)
        uek.unravel(new_buf, self.params['Object.vertex_positions_buf'])
        self.params.set_dirty('Object.vertex_positions_buf')
        self.params.update()

    def replace_mesh(self, verts, faces):
        """ new_verts, new_faces are enoki vectors or pytorch tensors """
        # transform torch tensors to enoki
        if isinstance(verts, torch.Tensor) and isinstance(faces, torch.Tensor):
            verts_ek = Vector3f(verts)
            if self.requires_grad:
                ek.set_requires_gradient(verts_ek, verts.requires_grad)
                self.params_optim = verts_ek
                self.params_optim_torch = verts
            faces_ek = UInt32(faces.flatten())
        elif isinstance(verts, Float32) and isinstance(faces, UInt32):
            verts_ek = uek.ravel(verts)
            faces_ek = faces
        elif isinstance(verts, Vector3f) and isinstance(faces, UInt32):
            verts_ek = verts
            faces_ek = faces
        else:
            raise ValueError("Check types of verts and faces.")

        # overwrite the vertex buf - we don't care about "inactive" vertices here
        uek.unravel(verts_ek, self.params["Object.vertex_positions_buf"])

        # 1..K need to be replaced by new faces
        self.params["Object.faces_buf"][:ek.slices(faces_ek)] = faces_ek[:]
        # K+1..N need to be set to 0
        if self.faces_size > ek.slices(faces_ek):
            self.params["Object.faces_buf"][ek.slices(faces_ek):] = \
                ek.zero(UInt32, self.faces_size - ek.slices(faces_ek))[:]

        self.params.set_dirty("Object.vertex_positions_buf")
        self.params.set_dirty("Object.faces_buf")
        self.params.update()

    def requires_grad_(self, requires:bool=True):
        self.requires_grad = requires
        if requires:
            verts_ek = uek.ravel(self.params['Object.vertex_positions_buf'])
            ek.set_requires_gradient(verts_ek, True)

    def render_scene(self, spp=None):
        if self.requires_grad:
            params_torch = {'vertices' : self.params_optim_torch}
            # call the scene's integrator to render the loaded scene
            rendered = render_torch(self.scene,
                                    spp=spp,
                                    **params_torch,
                                    )
        else:
            rendered = render_torch(self.scene, spp=spp)
        rendered = rendered.permute(2, 0, 1)

        # rendered has 8 channels: r-g-b-d(AOV)-r-g-b-a
        self.gray = rendered[0].unsqueeze(0) # one color channel
        self.alpha = rendered[-1].unsqueeze(0) # alpha channel

    def read_output(self, img_mode):
        if img_mode == 'silhouette':
            return self.read_silhouette()
        elif img_mode == 'shading':
            return self.read_shading()
        elif img_mode == 'mooney':
            img, blurred = self.make_mooney()
            self.blurred = blurred
            return img
        else:
            print('Not supported.')

    def read_shading(self):
        return self.gray

    def read_alpha(self):
        return self.alpha

    def read_silhouette(self):
        return 1-self.gray

    def make_mooney(self):
        # apply a Gaussian Filter on the whole image
        blurred = self.smoother(self.gray.unsqueeze(0), keep_dim=True).squeeze(0)

        if self.mooney_thresh_method == 'mean_mask':
            # mean of mask pixels
            mask = self.alpha.detach().bool()
            mooney_thresh = torch.mean(torch.masked_select(blurred.detach(), mask))
        elif self.mooney_thresh_method == 'otsu_mask':
            # otsu of mask pixels
            mask = self.alpha.detach().bool()
            mask_pixels = torch.masked_select(blurred.detach(), mask).cpu().numpy()
            mooney_thresh = threshold_otsu(mask_pixels)
        elif self.mooney_thresh_method == 'otsu_bbox':
            # otsu of bbox pixels
            rmin, rmax, cmin, cmax = bbox_mask(self.alpha.detach().bool())
            img = blurred.detach().cpu().numpy().squeeze()[rmin:rmax+1, cmin:cmax+1]
            mooney_thresh = threshold_otsu(img)
        else:
            print("Not implemented!")

        # differentiable threshold operation with binary mask as output
        # mooney = torch.relu(torch.tanh(1e3*(blurred-mooney_thresh)))
        mooney = torch.sigmoid(1e4*(blurred-mooney_thresh))

        # how many roi (mask or bbox) pixels are white?
        # white_ratio = torch.mean((mooney[roi_idx] > mooney_thresh).float()).detach().cpu().numpy()

        return mooney, blurred
