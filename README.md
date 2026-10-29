# Image_to_3DGS

Bien 1 video AI (nhan vat dung yen tren phong xanh, camera bay quanh) thanh 1 nhan vat **Gaussian Splat (3DGS)**
dung duoc trong shot VFX, chay tren Houdini 22 + GPU NVIDIA, Windows.

```
Electra ref video ──> Genjutsu (Higgsfield) ──> loc anh net ──> COLMAP ──> tach nen xanh ──> train 3DGS
                                                                                          └─> cat nguoi 3D
                                                    Fixer cac goc it quay <── render goc moi <─┘
                                                            └──> train lai ──> nhan vat GS cuoi
```

---

## 1. Phuong phap (vi sao lam the nay)

| Buoc | Cach lam | Ly do |
|---|---|---|
| Video tham chieu | Hinh nom Electra tren phong xanh co dau cong do, camera zig-zag tu thap len **cham tren dinh dau**, 3 giay cuoi vong ra sau lung, tay khep 15 do. 20 giay. | Shot boom-up nhin tu tren xuong, nen goc tren dinh dau quan trong nhat. Camera di deu theo **toc do xoay khung hinh** (khong theo quang duong) nen o dinh dau khong bi nhoe chuyen dong. |
| Video nhan vat | Genjutsu Lite (Higgsfield) chep chuyen dong camera tu video tham chieu, 6 anh tham chieu (4 goc trang phuc, chan dung, chan tran). | Nhan vat dung yen nhu tuong sap, nen GS hoc duoc. |
| Loc anh | `pick_frames.py`: chi giu anh **net** (Laplacian so voi cac khung xung quanh), moi anh **trung anh truoc 50-62 %** dien tich, du cac goc. 20 giay -> ~160 anh. | It anh hon nhung toan anh tot: COLMAP nhanh, khong co anh nhoe lam ban GS. |
| Dung camera | COLMAP (pycolmap) tren anh co ca phong (`--keep-bg`), chay o **720** roi nhan camera len kich thuoc goc. | Phong xanh + dau cong cho camera chinh xac nhat; 720 nhanh gap nhieu lan ma camera van dung. |
| Tach nen | Chroma key xanh trong Houdini COP, **tu do mau nen cua tung video**, AI (rembg) lam huong dan de giu vai/hoa van xanh. Mep mem 6 px. | Da thu train **khong tach nen**: dep o goc anh nhung khong cat duoc nguoi ra sach (ao trang lan vao nen xanh). Van can tach nen. |
| Train | Houdini 22 ML (gsplat) tren anh goc 1080, 50.000 buoc. | |
| Cat nguoi 3D | `carve_person.py`: giu splat nam trong mask o >= 85 % goc nhin, cat tru tru quanh nguoi, dung nguoi day cao 1.6 m. **Khong "got mep"** (an mat ban tay). | |
| Fixer | Render GS o 2 vong goc video it quay (quanh dinh dau 55/75 do; nua sau lung ngang tam mat), **NVIDIA Fixer** (Cosmos 0.6B) lam sach, cho vao train lai (kieu Difix3D+). Khong dua goc phia truoc vao (da quay du, Fixer chi lam mem mat). | Lung va 2 ben het manh vun 7 mau. |

---

## 2. Cai dat (1 lan tren may moi)

Can: Windows, GPU NVIDIA (da test RTX 24 GB), **Houdini 22** (co GSOPs, ban Indie duoc), **Python 3.13**
(tick "Add to PATH"), **Git for Windows** (de co `bash`), ffmpeg.

1. Clone repo:
   ```bash
   git clone https://github.com/Nghifx/Image_to_3DGS.git
   ```
2. Dat `ffmpeg.exe` vao `tools\ffmpeg\ffmpeg.exe` (hoac co ffmpeg tren PATH).
3. Double-click **`setup.bat`**: tao `tools\venv` (rembg, pycolmap, opencv), dang ky plugin Houdini.
4. NVIDIA Fixer (code + model khong nam trong repo nay):
   ```bash
   git clone https://github.com/nv-tlabs/Fixer.git tools/Fixer_src
   ```
   - Chep noi dung `tools/Fixer_src` vao `tools/Fixer` (giu lai `tools/Fixer/win_shims` va
     `tools/Fixer/requirements_fixer_win.txt` cua repo nay - win_shims thay transformer-engine/megatron
     vi khong build duoc tren Windows).
   - Tai model tu Hugging Face `nvidia/Fixer` (NVIDIA Open Model License, dung thuong mai duoc) vao:
     ```
     tools/Fixer/models/base/model_fast_tokenizer.pt
     tools/Fixer/models/base/tokenizer_fast.pth
     tools/Fixer/models/pretrained/pretrained_fixer.pkl
     ```
   - Double-click **`setup_fixer.bat`** (tao `tools\fixer_venv`, ~4 GB, torch cu128).
5. Lan train dau tien Houdini tu tao `ml\venv.gsplat.localscheduler` (30-60 phut, chi 1 lan).

Houdini o duong dan khac mac dinh: dat bien `HYTHON=".../bin/hython.exe"` truoc cac lenh.

---

## 3. Quy trinh 1 nhan vat

### Buoc 1 - Video tham chieu (co san)
`khmer_girl/ref_electra_arc_v030_20s/electra_arc_v030_480f.mp4`
Muon sua camera/tu the: `hython scripts/electra_ref_video.py` (doc ghi chu dau file).

### Buoc 2 - Video Genjutsu (Higgsfield)
- Model: Genjutsu Lite (`hf_mult_motion_control_flash`), 1080p.
- `@Video 1` = video tham chieu buoc 1.
- `@Image 1-4` = trang phuc (truoc, ben, sau, ben); `@Image 5` = chan dung; `@Image 6` = chan tran.
- Prompt ngan, chi dung tu khang dinh (tranh "no", "khong"), vi du:
  ```
  the camera uses a very fast shutter speed of 1/2000s, every frame crisp and sharp.
  She stands perfectly still like a wax figure, her hands rest still at her sides.
  ```
- Anh sang deu, khong ro nguon sang. Nhan vat di chan tran.

### Buoc 3 - Tu video ra GS (1 lenh, ~25-35 phut)
Dat video vao `chars/<ma>/video.mp4` (vd `chars/012/video.mp4`), mo Git Bash o thu muc repo:
```bash
MATTE=key bash scripts/batch_char.sh 012
```
Tu dong: loc anh -> COLMAP 720 -> do mau nen + tach nen -> train 1080 -> cat nguoi.
- It/nhieu anh hon: `PICK_MIN=0.4 PICK_MAX=0.55` (~75 anh), mac dinh 0.5/0.62 (~160 anh).
- Ket qua: `GSplats/c012/person/c012_person.ply` (+ `.json`: dung nguoi day 1.6 m).
- Nhat ky: `chars/012/batch.log`. Chay lai lenh se tiep tuc tu buoc dang do.

### Buoc 4 - Fixer cac goc it quay (~20 phut)
```bash
bash scripts/batch_fixer.sh 012
```
Ket qua dung cho shot: `GSplats/c012x/person/c012x_person.ply`. Playblast kiem tra: `chars/012/pb_c012x/`.

### Buoc 5 - Kiem tra / dua vao shot
```bash
ELECTRA_PATH=slow hython scripts/gs_electra_playblast.py GSplats/c012x/person/c012x_person <thu_muc_ra>
```
Trong Houdini: file ply qua `bakegsplat`, xoay `+90 do quanh X` (undo COLMAP), roi
`P' = scale * R (P - foot)` voi `R, foot, scale` trong file `.json` -> nguoi dung tai goc toa do, cao 1.6 m, mat huong +Z.

---

## 4. Cau truc thu muc

| Duong dan | Noi dung |
|---|---|
| `scripts/pick_frames.py` | loc anh net, trung 30-60 % |
| `scripts/batch_char.sh` | buoc 3 (goi cac script ben duoi) |
| `scripts/batch_dataset.py` | dataset raw/guide/screen/final |
| `scripts/gs_colmap_ai.py`, `orbit_solve.py` | COLMAP |
| `scripts/batch_grade.py` | tach nen xanh trong Houdini COP |
| `scripts/h22_train_gsplats.py` | train 3DGS bang Houdini 22 |
| `scripts/carve_person.py` | cat nguoi 3D, dung thang |
| `scripts/batch_fixer.sh`, `gs_novel_views.py`, `fixer_run.py`, `gs_add_views.py` | buoc 4 Fixer |
| `scripts/electra_ref_video.py`, `gs_electra_playblast.py` | video tham chieu, playblast kiem tra |
| `houdini/gsplat_studio/` | plugin Houdini (panel 4 buoc) |
| `houdini/electra_*.hip` | canh Electra cho video tham chieu / playblast |
| `tools/Fixer/win_shims/` | thay the thu vien Fixer khong build duoc tren Windows |

Thu muc sinh ra khi chay (khong dua len git): `tools/venv`, `tools/fixer_venv`, `ml/`, `chars/`, `data/`,
`colmap/`, `GSplats/`.

---

## 5. Loi thuong gap

- **Lung / 2 ben lem o goc ngang tam mat**: video khong quay goc do -> chay buoc 4 (Fixer).
- **Mat ban tay**: `carve_person.py` phai de `CARVE_EDGE_K=0` (mac dinh).
- **Vien xanh quanh nguoi**: xoa `chars/<ma>/screen_rgb.json` de do lai mau nen roi chay lai buoc 3.
- **Mat chi tiet khi nhin ngang**: GS chi dep o goc co trong video. Kiem tra bang
  `ml/venv.gsplat.localscheduler/Scripts/python.exe scripts/gs_render_check.py <run> <ply>` (so anh goc vs GS).
- **Genjutsu khong theo duong camera**: video tham chieu dai qua gioi han model, hoac prompt dai qua - giu prompt ngan.

---

Tai lieu chi tiet hon cho plugin Houdini: `SETUP.md`, `houdini/gsplat_studio/README.md`.
