import sys as _sys
globals().update({k: v for k, v in vars(_sys.modules["emerge._emerge.geo.pmlbox"]).items() if not k.startswith("__")})

import numpy as np
PROFILE = None
def _add_pml_layer(center: tuple[float, float, float],
                   dims: tuple[float, float, float],
                   direction: tuple[float, float, float],
                   thickness: float,
                   Nlayers: int,
                   N_mesh_layers: int,
                   exponent: float,
                   deltamax: float,
                   material: Material) -> GeoVolume:
    
    if material.frequency_dependent:
        raise ValueError('EMerge cannot handle frequency dependent material properties for PML layers at this point.')
    if material.coordinate_dependent:
        raise ValueError('Its not possible to define PML regions for materials that are coordinate dependent.')
    
    mater = material.er.scalar(1e9)
    matur = material.ur.scalar(1e9)
    
    px, py, pz = center
    W,D,H = dims
    dx, dy, dz = direction

    pml_block_size = [W, D, H]
    new_center = [px, py, pz]
    
    sxf = lambda x, y, z: np.ones_like(x, dtype=np.complex128)
    syf = lambda x, y, z: np.ones_like(x, dtype=np.complex128)
    szf = lambda x, y, z: np.ones_like(x, dtype=np.complex128)
    
    p0x = px + dx*(W/2+thickness/2)
    p0y = py + dy*(D/2+thickness/2)
    p0z = pz + dz*(H/2+thickness/2)

    tW, tD, tH = W, D, H

    _namesuffix = ''
    if dx != 0:
        tW = thickness
        pml_block_size[0] = thickness
        new_center[0] = new_center[0] + (W/2 + thickness/2) * dx
        def sxf(x, y, z):
            return PROFILE((dx*(x-px-(dx*W/2)) / thickness), exponent, deltamax)
        _namesuffix += f'{_sgn(dx)}x'
    if dy != 0:
        tD = thickness
        pml_block_size[1] = thickness
        new_center[1] = new_center[1] + (D/2 + thickness/2) * dy
        def syf(x, y, z):
            return PROFILE((dy*(y-py-(dy*D/2)) / thickness), exponent, deltamax)
        _namesuffix += f'{_sgn(dy)}y'
    if dz != 0:
        tH = thickness
        pml_block_size[2] = thickness
        new_center[2] = new_center[2] + (H/2 + thickness/2) * dz
        def szf(x, y, z):
            return PROFILE((dz*(z-pz-(dz*H/2)) / thickness), exponent, deltamax)
        _namesuffix += f'{_sgn(dz)}z'
    def ermat(x, y, z):
        ers = np.zeros((3,3,x.shape[0]), dtype=np.complex128)
        ers[0,0,:] = mater * syf(x,y,z)*szf(x,y,z)/sxf(x,y,z)
        ers[1,1,:] = mater * szf(x,y,z)*sxf(x,y,z)/syf(x,y,z)
        ers[2,2,:] = mater * sxf(x,y,z)*syf(x,y,z)/szf(x,y,z)
        return ers
    
    def urmat(x, y, z):
        urs = np.zeros((3,3,x.shape[0]), dtype=np.complex128)
        urs[0,0,:] = matur * syf(x,y,z)*szf(x,y,z)/sxf(x,y,z)
        urs[1,1,:] = matur * szf(x,y,z)*sxf(x,y,z)/syf(x,y,z)
        urs[2,2,:] = matur * sxf(x,y,z)*syf(x,y,z)/szf(x,y,z)
        return urs
    
    pml_box = Box(*pml_block_size, new_center, alignment=Alignment.CENTER)
    pml_box._unset_constraints = True
    
    planes = []
    thl = thickness / Nlayers
    planes = []

    if dx != 0:
        ax1 = np.array([0, tD, 0])
        ax2 = np.array([0, 0, tH])
        for n in range(Nlayers-1):
            plate = Plate(np.array([p0x-dx*thickness/2 + dx*(n+1)*thl, p0y-tD/2, p0z-tH/2]), ax1, ax2)
            planes.append(plate)
    if dy != 0:
        ax1 = np.array([tW, 0, 0])
        ax2 = np.array([0, 0, tH])
        for n in range(Nlayers-1):
            plate = Plate(np.array([p0x-tW/2, p0y-dy*thickness/2 + dy*(n+1)*thl, p0z-tH/2]), ax1, ax2)
            planes.append(plate)
    if dz != 0:
        ax1 = np.array([tW, 0, 0])
        ax2 = np.array([0, tD, 0])
        for n in range(Nlayers-1):
            plate = Plate(np.array([p0x-tW/2, p0y-tD/2, p0z-dz*thickness/2 + dz*(n+1)*thl]), ax1, ax2)
            planes.append(plate)
    
    erfunc = CoordDependent(max_value=mater, matrix=ermat)
    urfunc = CoordDependent(max_value=matur, matrix=urmat)

    pml_box.properties += Material(er=erfunc, ur=urfunc,_neff=np.sqrt(mater*matur), color='#bbbbff', opacity=0.1, name=f'PML_mat[{_namesuffix}]')
    pml_box.max_meshsize = thickness/N_mesh_layers
    pml_box._embeddings = planes
    
    return pml_box


