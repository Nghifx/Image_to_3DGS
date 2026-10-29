# GSplat Studio - cai tren may moi

Video 360 -> Gaussian splat trong Houdini 22 (4 buoc: nhap video, tach nen, COLMAP, train).

## Can co san tren may

- **Houdini 22.0** (ban thuong mai: khong dong dau, render full do phan giai)
- **Python 3.13** tu python.org (tick "Add python.exe to PATH" khi cai)
- Card NVIDIA (train splat va COLMAP chay GPU)

## Cai dat (mot lan)

1. Giai nen goi nay vao mot thu muc bat ky, vd `D:\Gaussian_Splatting`
   (duong dan nen khong dau, khong khoang trang).
2. Bam dup **setup.bat**. Mat 5-10 phut (tai goi Python).
3. Mo Houdini 22 > **+** > **New Pane Tab Type** > **GSplat Studio**.

Lan train dau tien Houdini tu cai moi truong hoc may cua no (~30-60 phut, chi mot lan).

## Trong goi co gi

| Thu muc | Noi dung |
|---|---|
| `scripts/` | cac buoc xu ly (tach nen, COLMAP, train, render tham chieu) |
| `houdini/gsplat_studio/` | plugin panel GSplat Studio |
| `houdini/*.hip*` | scene mau, vd `electra_orbit_ref.hip` (Electra T-pose + camera quay 360) |
| `khmer_girl/`, `ghost_wraith/` | hinh/video goc da tao bang Higgsfield |
| `houdini/video/Ghost_01/` | 240 anh goc cua con ma |
| `data/<ten>/` | anh da tach nen (buoc 2) |
| `colmap/<ten>/` | camera da giai (buoc 3) - san sang train |
| `GSplats/khmer_01/` | splat da train cua co gai Khmer |
| `tools/ffmpeg`, `tools/models` | ffmpeg va model tach nen, dung san khong can tai |

## Viec nen lam dau tien tren may co ban quyen

- Render lai video tham chieu khong dong dau (1920x1080):
  `"C:\Program Files\Side Effects Software\Houdini 22.0.xxx\bin\hython.exe" scripts\gs_orbit_reference.py`
- Luong "GSplat Studio - Object" tu render 2048x2048 khi khong phai ban Apprentice.

## Khong co trong goi (tu tao lai)

`tools\venv` (setup.bat tao), moi truong train cua Houdini (tu tao lan dau),
du an Wukong cu (file goc nam o o mang N:).
