"""Concise supervisor summary (Macedonian, A4, 2-3 pages) as PDF, built with reportlab.

Numbers are copied from REPORT_MK.md / the evaluation CSVs. The PDF contains no file paths and no examination
identifiers (only the labels E1-E7); its metadata are set explicitly. After building, the script checks
page count, metadata and the extracted text for paths and identifiers.

Run with any Python that has reportlab and pypdf (here a throw-away venv outside the project):
    /tmp/manu_pdfenv/venv/bin/python evaluation/visual_analysis_and_report/make_pdf.py
"""
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
OUT = HERE / 'MANU_supervisor_summary_MK.pdf'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DejaVu', str(FONT_DIR / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', str(FONT_DIR / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold')

INK, MUTED, RULE, SHADE = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#d6d5d0'), colors.HexColor('#f3f2ee')
BODY = ParagraphStyle('body', fontName='DejaVu', fontSize=8.6, leading=11.4, textColor=INK, alignment=TA_LEFT, spaceAfter=3)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=7.2, leading=9.2, textColor=MUTED)
H1 = ParagraphStyle('h1', parent=BODY, fontName='DejaVu-Bold', fontSize=14, leading=17, spaceAfter=2)
H2 = ParagraphStyle('h2', parent=BODY, fontName='DejaVu-Bold', fontSize=10, leading=13, spaceBefore=6, spaceAfter=3)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=7.4, leading=9.0, spaceAfter=0)
CELLB = ParagraphStyle('cellb', parent=CELL, fontName='DejaVu-Bold')
BOX = ParagraphStyle('box', parent=BODY, fontSize=8.2, leading=10.6, backColor=SHADE, borderPadding=(5, 6, 5, 6),
                     spaceBefore=4, spaceAfter=8)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BODY):
    return [Paragraph(f'•&nbsp;&nbsp;{t}', ParagraphStyle('b', parent=style, leftIndent=10, firstLineIndent=-8)) for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK),
             ('LINEBELOW', (0, 1), (-1, -1), 0.25, RULE), ('TOPPADDING', (0, 0), (-1, -1), 2.2),
             ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]
    t.setStyle(TableStyle(style))
    return t


def figure(name, width_cm, caption):
    path = HERE / 'figures' / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return KeepTogether([img, Spacer(1, 2), p(caption, SMALL)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVu', 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, 'MANU · лимфомска сегментација на FDG PET/CT · резиме за ментор · 19.10.2026')
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'{doc.page}')
    canvas.restoreState()


def story():
    s = [p('MANU: детекција и сегментација на лимфомски лезии на FDG PET/CT', H1),
         p('Кратко резиме за проф. Игор Мишковски · завршени чекори 1–3 · 19.10.2026', SMALL), Spacer(1, 6)]

    s += [p('1. Цел и статус', H2),
          p('Цел: автоматска детекција и сегментација на лимфомски лезии и локална евалуација со Dice, false-positive/false-negative '
            'волумен (FPV/FNV), детекција по лезија, TMTV и SUVmax. Завршено: (1) издвојување на лимфомските случаи од autoPET, '
            '(2) два експерименти со 5-fold cross-validation по пациент, (3) ориентациска споредба со pretrained autoPET III. '
            'Ниту еден резултат не е добиен со повторно тренирање за овој извештај; сите бројки се од зачуваните евалуации.')]

    s += [p('2. Податоци и експерименти', H2),
          p('autoPET (FDG-PET-CT-Lesions), испитувања со дијагноза LYMPHOMA: <b>145 испитувања од 144 пациенти</b>, сите со барем '
            'една анотирана лезија (нема негативни случаи). Двата експерименти користат иста residual 3D U-Net (MONAI, CT + PET влез).'),
          table([['', 'A: Microsoft baseline', 'B: testTraining'],
                 ['Код', 'Microsoft lymphoma-segmentation-dnn, непроменет', 'иста мрежа и рецепт, изменет протокол'],
                 ['Patch', '192³', '64³'],
                 ['Епохи', '10 (cosine до 0)', 'буџет 500, early stopping: <b>реално 112–272</b> (избрани 62–222)'],
                 ['Валидација / избор', 'тест fold-от (checkpoint-selection leakage; тука без ефект, избрана последна епоха)',
                  'внатрешна валидација, 20% од тренинг-пациентите'],
                 ['Fold поделба', 'по пациент, seed 42', 'по пациент, seed 20261006 (различни folds)']],
                [3.0, 6.6, 7.4]),
          Spacer(1, 3),
          p('Поради различните поделби, валидна е само спарената споредба по испитување (секое испитување е предвидено од модел '
            'што не го видел пациентот). Метриките се пресметани по дефинициите на Microsoft (18-connectivity; проверено 176/176 '
            'совпаѓања со оригиналните функции).', SMALL)]

    s += [p('3. Резултати: A наспроти B на сите 145 испитувања', H2),
          table([['Метрика', 'A', 'B', 'B − A (95% CI), Holm p'],
                 ['Dice, mean (median)', '0,301 (0,267)', '0,277 (0,259)', '−0,025 (−0,049 до −0,001), p = 0,51'],
                 ['FPV, median ml', '471', '291', '−306 ml, p &lt; 10⁻⁸; B понизок кај 100/145'],
                 ['FNV, median ml', '0,49', '1,8', '+7,8 ml, p &lt; 10⁻⁹; A понизок кај 77, B кај 13'],
                 ['TMTV апс. грешка, mean ml', '752', '330', '−422 ml, p &lt; 10⁻¹¹'],
                 ['SUVmax апс. грешка, mean', '51,1', '51,6', '+0,5, p = 1,0'],
                 ['Lesion recall C1 (pooled)', '0,49', '0,41', 'по испитување −0,09, p &lt; 10⁻⁸'],
                 ['Lesion precision / F1 C1 (pooled)', '0,09 / 0,16', '0,29 / 0,34', '—'],
                 ['FP лезии по испитување, mean', '124,5', '33,6', '−91, p &lt; 10⁻¹²']],
                [4.6, 2.4, 2.4, 7.6]),
          Spacer(1, 3),
          p('<b>Dice не ги разликува моделите; другите метрики покажуваат trade-off.</b> A е посензитивен (повеќе пронајдени лезии, '
            'помал FNV), B е поспецифичен (≈3,7× помалку FP лезии, половина FPV, под половина TMTV грешка). Двата сериозно '
            'пресегментираат (предвиден TMTV во медијана 4,7× и 2,4× од референтниот). SUVmax е погрешен и кај двата: предвидениот '
            'SUVmax е поголем од референтниот кај 129 (A) и 127 (B) од 145 испитувања. p-вредностите ја одразуваат само '
            'варијабилноста на случаите, не и на тренингот (по едно тренирање).')]

    s += [p('4. autoPET III на 10 испитувања (ориентациски)', H2),
          p('<b>Изјава:</b> pretrained autoPET III checkpoint-от е трениран на autoPET податоци што <b>ги вклучуваат овие 10 испитувања</b>. '
            'Резултатите се оптимистичка, не-независна референца, а не мерка за генерализација; поради n = 10 не се наведува '
            'статистичка значајност.', BOX),
          table([['10 испитувања', 'A', 'B', 'C: autoPET III*'],
                 ['Dice, mean (опсег)', '0,35 (0,01–0,75)', '0,33 (0,02–0,65)', '0,92 (0,85–0,98)'],
                 ['FPV / FNV, median ml', '590 / 0,42', '184 / 1,3', '0,14 / 0,11'],
                 ['TMTV апс. грешка, median ml', '626', '259', '10'],
                 ['SUVmax еднаков на референтниот', '1/10', '1/10', '8/10'],
                 ['Lesion recall / precision C1', '0,52 / 0,13', '0,39 / 0,45', '0,74 / 0,89']],
                [5.6, 3.6, 3.6, 4.2]),
          p('* трениран на податоци што ги вклучуваат овие испитувања.', SMALL)]

    s += [PageBreak(), p('5. Визуелна анализа на грешките', H2),
          p('Седум информативни испитувања (E1–E7) се прикажани со CT, PET, референца и предвидувањата на A, B и C, на пресеци со најголема '
            'референтна површина и најголема FP површина. Пред цртање, за сите 60 слики се проверени големината, spacing, origin '
            'и целосната direction матрица (сите на иста мрежа, LAS); ориентацијата е потврдена на CT. Секоја FP компонента ≥ 20 ml на '
            'A и B (48) е прегледана на CT и PET; структура е именувана само кога е јасно препознатлива (визуелна проценка, не органска '
            'сегментација и не радиолошки наод).'),
          figure('fig09_fp_anatomy.png', 13.5,
                 'Слика 1. Удел на целиот FP волумен по визуелно идентификувана структура (10 испитувања). autoPET III има вкупно 9,4 ml FP.'),
          Spacer(1, 4),
          *bullets([
              '<b>Околу 93% (A) и 86% (B) од FP волуменот лежи во мозокот, мочниот меур, срцето (миокард на лева комора) и бубрезите</b>, '
              'органи со нормално високо FDG навлегување или излачување.',
              '<b>Изворот на погрешниот SUVmax:</b> вокселот со предвидениот SUVmax лежи надвор од референцата кај 9/10 испитувања за A и B '
              '(2/10 за C); кај прегледаните компоненти тоа е мочниот меур или мозокот, не тумор.',
              '<b>Мали тумори:</b> кај E6 (6,7 ml) FP волуменот е 15–70× поголем од туморот, па Dice на A/B е ≈ 0 (C: 0,913).',
              '<b>Големи тумори:</b> кај E1 (2481 ml) B ја подсегментира големата абдоминална болест и пропушта 94 мали лезии; A наоѓа повеќе, '
              'но го означува речиси целиот мозок (1019 ml FP). Ова е конзистентно со губење контекст кај 64³ patch, но не е докажано.',
              '<b>Мали лезии</b> се пропуштаат од сите три модели, дури и од C (кај E2: A 76, B 90, C 50 пропуштени лезии).',
              'FNV = 0 не значи целосна сегментација: лезија допрена со еден воксел не се брои во FNV (E7).'])]

    s += [figure('fig05_E4.png', 17.0,
                 'Слика 2. Пример E4 (референтен TMTV 285 ml, 17 лезии): CT, PET, референца и предвидувањата на A, B и C '
                 '(аква = TP, портокалово = FP, виолетово = FN, по воксел). Горе: абдоминалната болест ја наоѓаат сите три модели. '
                 'Средина: FP на A и B во миокардот на левата комора. Долу (коронарно): FP во мозокот (A, B) и мочниот меур (A). '
                 'Сите 7 примери и MIP прегледот се во папката со слики. * autoPET III е трениран на ова испитување.')]

    s += [p('6. Главни ограничувања', H2),
          *bullets([
              'Checkpoint-selection leakage во Microsoft кодот (A); тука без ефект, но процедурата е погрешна за подолги тренинзи.',
              'A и B се разликуваат во повеќе фактори одеднаш (patch, crop, тренинг-пациенти, избор на checkpoint, запирање, прецизност) и '
              'во fold поделбата; разликите не можат да се припишат на една причина.',
              'Нема повторени тренинзи (варијансата на тренингот е непозната); A има само 10 епохи; B е запрен со висок learning rate.',
              'SUVmax е една вредност по испитување, па еден FP во меурот или мозокот ја прави погрешна целата мерка.',
              'autoPET III не е независен; анатомската проценка е визуелна; кохортата нема негативни испитувања.'], BODY)]

    s += [p('7. Предлози за следни чекори (не се завршена работа)', H2),
          *bullets([
              '<b>Поправен baseline:</b> подолг распоред (100–500 епохи) со избор на checkpoint без тест fold-от и една заедничка split датотека; 3 seeds по конфигурација.',
              '<b>Намалување на физиолошките FP</b>, мерено со FPV, TMTV и SUVmax: постпроцесирање со органски маски (мозок, меур, бубрези, срце), '
              'тренинг со негативни autoPET испитувања, или органски-свесен auxiliary output; секое како посебен контролиран експеримент.',
              '<b>Фер споредба со силен модел:</b> autoPET III на надворешни лимфомски испитувања, или nnU-Net трениран на нашите folds.',
              '<b>GenAI (за дискусија):</b> diffusion-базирана синтеза на мали лезии за аугментација; promptable foundation модели за '
              'интерактивна корекција на FP; модел на нормално FDG навлегување за детекција на аномалии; јазичен модел за нацрт '
              'структуриран извештај (TMTV, SUVmax, локализација) со задолжителна проверка од лекар.'])]
    return s


def check(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    meta = {k: str(v) for k, v in (reader.metadata or {}).items()}
    patterns = [r'/home', r'/mnt', r'[A-Za-z]:\\', r'eva25', r'PETCT_', r'Users', r'\.py\b', r'\.csv\b']
    hits = {pat: re.findall(pat, text + ' '.join(meta.values())) for pat in patterns}
    hits = {k: v for k, v in hits.items() if v}
    print(f'pages: {len(reader.pages)}; metadata: {meta}')
    print('sensitive-pattern hits:', hits or 'none')
    if hits or not 2 <= len(reader.pages) <= 3:
        raise SystemExit('PDF check failed')


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                            title='MANU: резиме за ментор', author='MANU проект', subject='Лимфомска сегментација на FDG PET/CT',
                            creator='MANU проект', keywords='', invariant=1)
    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    check(OUT)
    print('written', OUT.name)


if __name__ == '__main__':
    main()
