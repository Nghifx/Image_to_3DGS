# GSplat Studio

Panel trong Houdini 22: bien mot video quay 360 thanh Gaussian splat qua 4 buoc,
moi buoc co thanh tien do rieng va ket qua hien ngay.

## Cai dat

Da cai san qua file package:
`C:\Users\<user>\Documents\houdini22.0\packages\gsplat_studio.json`

Mo panel: menu **+ > New Pane Tab Type > GSplat Studio**.
(Luong cu "render object trong canh" van con, o **GSplat Studio - Object**.)

## 4 buoc

| Buoc | Lam gi | Ket qua |
|---|---|---|
| 1 Nhap video | Chon video (hoac thu muc anh). Panel hien ngay so khung, kich thuoc, fps. Chon **Buoc nhay**: 1 = lay het, 2 = lay 1 bo 1 (nhu Increment cua ROP). | so anh se xuat |
| 2 Tach nen | ffmpeg xuat anh PNG khong nen o do phan giai goc, tach nen, cat khung quanh nhan vat. | `data/<ten>/` |
| 3 COLMAP | COLMAP tu tinh vi tri camera tu anh (khoa ong kinh). Neu giai duoc duoi 80% so anh thi tu chuyen sang dat camera len vong tron hoan hao theo goc do. | `colmap/<ten>/` |
| 4 Train GSplats | ML Train GSplats cua Houdini. | `GSplats/<ten>/` |

- **Chay het**: chay tu buoc chua xong toi het. Nut **▶** o moi dong: chay rieng buoc do.
- Doi ten nhan vat: panel tu danh dau buoc nao da co ket qua.
- Khung xem trong panel: buoc 2 hien anh vua tach nen, buoc 3 hien anh kiem tra (cham xanh = camera khop).
- Viewport Houdini: buoc 3 hien diem 3D + vong camera (`/obj/GS_COLMAP`), buoc 4 hien splat dang hinh thanh (`/obj/GS_PREVIEW`).
- **Tuy chon** (an mac dinh): chieu cao nhan vat, so buoc train, Claude kiem tra (tat).

## Claude kiem tra goc (thu nghiem, tat mac dinh)

Da thu ngay 2026-10-22: Claude xac nhan dung bo camera tot, nhung bo sot mot
loi lech 12 do co y cai vao - ca khi xem anh render tu splat. Nen buoc 3 chi
dung COLMAP do goc (khong ton phi). Code van giu: bat o **Tuy chon** neu muon
thu; can `claude` -> `/login` mot lan, ton khoang 1-3 USD moi nhan vat.

Cong cu chan doan kem theo: `scripts/gs_render_check.py` render splat tu tung
camera va cham diem voi anh goc (chay bang python cua trainer:
`ml\venv.gsplat.localscheduler\Scripts\python.exe`).

## Cau truc

```
panel.py         giao dien 4 buoc
panel_object.py  luong cu: render object trong canh roi train
runner.py        chay tung buoc bang tien trinh rieng, doc tien do
preview.py       nap diem COLMAP va splat vao viewport
config.py        duong dan va gia tri mac dinh
```

Script nam o `D:\Gaussian_Splatting\scripts`:
`gs_probe.py` (buoc 1), `gs_frames.py` (buoc 2), `gs_colmap_ai.py` + `orbit_solve.py`
(buoc 3), `h22_train_gsplats.py` (buoc 4).
