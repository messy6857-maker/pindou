"""Tiny voxel renderer (painter's algorithm). col[y,x,z]: y down, x right, z=0 front; -1 empty, -2 support."""
import numpy as np
from PIL import Image, ImageDraw
PAL_RGB=None          # set by caller: array (n,3)
FILLRGB=(40,230,60)
def rgb_of(c): return FILLRGB if c==-2 else tuple(int(v) for v in PAL_RGB[c])
def render(col, yaw_deg, pitch_deg=20, s=12, bg=(255,255,255,0)):
    """col[y,x,z]; y down, z=0 front. Camera looks from front (-z) rotated by yaw around vertical."""
    occ=col!=-1
    Y,X,Z=col.shape
    yaw=np.radians(yaw_deg); pit=np.radians(pitch_deg)
    # world coords: x right, y up, z toward viewer (front = +z_world)
    cx,cy,cz=X/2,Y/2,Z/2
    def proj(p):
        x,y,z=p[...,0]-cx,p[...,1]-cy,p[...,2]-cz
        xr=x*np.cos(yaw)+z*np.sin(yaw); zr=-x*np.sin(yaw)+z*np.cos(yaw)
        yr=y*np.cos(pit)-zr*np.sin(pit); zr2=y*np.sin(pit)+zr*np.cos(pit)
        return np.stack([xr,yr,zr2],-1)
    faces=[]
    dirs=[((1,0,0),'x+'),((-1,0,0),'x-'),((0,1,0),'y+'),((0,-1,0),'y-'),((0,0,1),'z+'),((0,0,-1),'z-')]
    light=np.array([0.4,0.7,0.6]); light/=np.linalg.norm(light)
    ys,xs,zs=np.where(occ)
    for y,x,z in zip(ys,xs,zs):
        wx,wy,wz=x,Y-1-y,Z-1-z   # world
        for (dx,dy,dz),_ in dirs:
            # neighbor in grid coords
            nx_,ny_,nz_=x+dx,y-dy,z-dz
            if 0<=nx_<X and 0<=ny_<Y and 0<=nz_<Z and occ[ny_,nx_,nz_]: continue
            n=proj(np.array([[dx,dy,dz]],float)+np.array([cx,cy,cz]))[0]
            if n[2]<=1e-6: continue
            # face corners
            c=np.array([wx+0.5,wy+0.5,wz+0.5])+0.5*np.array([dx,dy,dz])
            ax=[i for i in range(3) if (dx,dy,dz)[i]==0]
            corners=[]
            for sa,sb in ((-1,-1),(1,-1),(1,1),(-1,1)):
                p=c.copy(); p[ax[0]]+=0.5*sa; p[ax[1]]+=0.5*sb; corners.append(p)
            pc=proj(np.array(corners))
            shade=0.55+0.45*max(0,float(np.dot(n,light)))
            rgb=np.array(rgb_of(col[y,x,z]))*shade
            faces.append((pc[:,2].mean(),pc[:,:2],tuple(int(min(255,v)) for v in rgb)))
    faces.sort(key=lambda f:f[0])
    allp=np.concatenate([f[1] for f in faces])
    mn=allp.min(0); mx=allp.max(0); pad=1
    W=int((mx[0]-mn[0]+2*pad)*s); Hh=int((mx[1]-mn[1]+2*pad)*s)
    img=Image.new('RGBA',(W,Hh),bg); d=ImageDraw.Draw(img)
    for _,p2,rgb in faces:
        pts=[((px-mn[0]+pad)*s, Hh-(py-mn[1]+pad)*s) for px,py in p2]
        d.polygon(pts,fill=rgb,outline=tuple(max(0,int(v*0.8)) for v in rgb))
    return img
