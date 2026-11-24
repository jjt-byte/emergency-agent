#!/usr/bin/env python3
"""Build the Chinese PDF locally with system Cairo/Pango; no network/browser."""
import html
from pathlib import Path
import re
import sys

try:
    import cairo
    import gi
    gi.require_version('Pango', '1.0')
    gi.require_version('PangoCairo', '1.0')
    from gi.repository import Pango, PangoCairo
except ImportError as exc:
    raise SystemExit('Use system python3; install python3-cairo python3-gi gir1.2-pango-1.0 fonts-noto-cjk') from exc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/技术白皮书.md'
OUTPUT = ROOT/'docs/技术白皮书.pdf'
W, H = 595.276, 841.890
LEFT, RIGHT, TOP, BOTTOM = 48, 48, 65, 60
BLUE, INK, MUTED = (0.08, 0.29, 0.51), (0.12, 0.17, 0.23), (0.39, 0.45, 0.51)


def tokens(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('# '):
            yield ('chapter', line[2:])
        elif line.startswith('## '):
            yield ('heading', line[3:])
        elif line == '<!-- architecture -->':
            yield ('diagram', '')
        elif line.startswith('```'):
            block=[]
            i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]);i+=1
            yield ('code','\n'.join(block))
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in row):rows.append(row)
                i+=1
            yield ('table',rows)
            continue
        else:
            block=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','```','<!--')):
                i+=1;block.append(lines[i])
            yield ('paragraph',' '.join(block))
        i+=1


class Book:
    def __init__(self, surface):
        self.surface=surface
        self.c=cairo.Context(surface)
        self.page=0
        self.y=TOP
        self.chapter=''
        self.chapters={}
        self.started=False

    def box(self,x,y,w,h,color):
        self.c.set_source_rgb(*color);self.c.rectangle(x,y,w,h);self.c.fill()

    def layout(self,text,width,size=10.2,bold=False,mono=False):
        layout=PangoCairo.create_layout(self.c)
        PangoCairo.context_set_resolution(layout.get_context(),72)
        desc=Pango.FontDescription('Noto Sans CJK SC' if not mono else 'DejaVu Sans Mono')
        desc.set_absolute_size(size*Pango.SCALE)
        if bold:desc.set_weight(Pango.Weight.BOLD)
        layout.set_font_description(desc)
        layout.set_width(int(width*Pango.SCALE))
        layout.set_wrap(Pango.WrapMode.WORD_CHAR)
        layout.set_spacing(int(2.8*Pango.SCALE))
        markup=html.escape(text)
        markup=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',markup)
        markup=re.sub(r'`([^`]+)`',r'<span foreground="#174a78">\1</span>',markup)
        layout.set_markup(markup,-1)
        return layout

    def text(self,text,x,y,width,size=10.2,bold=False,color=INK,mono=False):
        layout=self.layout(text,width,size,bold,mono)
        self.c.set_source_rgb(*color);self.c.move_to(x,y);PangoCairo.show_layout(self.c,layout)
        return layout.get_pixel_size()[1]

    def finish_page(self):
        if self.page>1:
            self.box(LEFT,H-43,W-LEFT-RIGHT,0.6,(0.82,0.86,0.9))
            self.text('应急先行 · 原版 8080 技术白皮书',LEFT,H-34,380,8,color=MUTED)
            self.text(str(self.page),W-RIGHT-25,H-34,25,8,color=MUTED)
        self.c.show_page()

    def new_page(self,cover=False):
        if self.started:self.finish_page()
        self.started=True;self.page+=1;self.y=TOP
        self.box(0,0,W,H,(1,1,1))
        if not cover:
            self.box(LEFT,35,24,3,BLUE)
            self.text(self.chapter or '技术资料',LEFT+34,27,W-LEFT-RIGHT-34,8,color=MUTED)

    def ensure(self,height):
        if self.y+height>H-BOTTOM:self.new_page()

    def paragraph(self,text,size=10.2,mono=False):
        # Normal paragraphs fit a page; code is split into individual lines.
        layout=self.layout(text,W-LEFT-RIGHT,size,mono=mono)
        height=layout.get_pixel_size()[1]
        self.ensure(height+10)
        self.text(text,LEFT,self.y,W-LEFT-RIGHT,size,mono=mono)
        self.y+=height+11

    def table(self,rows):
        n=len(rows[0]);width=W-LEFT-RIGHT
        weights={2:[.36,.64],3:[.26,.36,.38],4:[.24,.25,.24,.27],5:[.27,.17,.14,.14,.28]}.get(n,[1/n]*n)
        compact = rows[0][0] == '编号'
        if compact: weights=[.10,.58,.32]
        col=[width*w for w in weights]
        padding = 3.5 if compact else 6
        def draw(row,header=False):
            heights=[self.layout(cell,col[i]-12,9.1,header).get_pixel_size()[1] for i,cell in enumerate(row)]
            height=max(heights)+padding*2
            self.ensure(height)
            self.box(LEFT,self.y,width,height,(0.08,.29,.51) if header else (.95,.97,.985))
            x=LEFT
            for i,cell in enumerate(row):
                self.text(cell,x+6,self.y+padding,col[i]-12,9.1,header,(1,1,1) if header else INK)
                x+=col[i]
            self.y+=height+1.5
        draw(rows[0],True)
        for row in rows[1:]:
            estimate=max(self.layout(c,col[i]-12,9.1).get_pixel_size()[1] for i,c in enumerate(row))+16
            if self.y+estimate>H-BOTTOM:
                self.new_page();draw(rows[0],True)
            draw(row)
        self.y+=12

    def diagram(self):
        height=195;self.ensure(height)
        labels=[('本地网页 / CLI','图片提问、视频导入、人工确认'),
                ('任务与证据控制','Agent 工具规划 · 风险规则 · 本地知识'),
                ('计算后端','CPU: Qwen + DAMO  |  NPU: 可选人员候选'),
                ('本地数据','报告、事件、截图、哈希与检测状态')]
        for index,(title,detail) in enumerate(labels):
            yy=self.y+index*47
            self.box(LEFT+12,yy,W-LEFT-RIGHT-24,39,(.92,.96,.985))
            self.text(title,LEFT+22,yy+8,116,10,True,color=BLUE)
            self.text(detail,LEFT+146,yy+8,W-LEFT-RIGHT-178,9.5)
            if index<3:
                self.text('↓',W/2-5,yy+37,20,9,color=BLUE)
        self.y+=height

    def cover(self):
        self.new_page(True)
        self.box(0,0,W,16,BLUE)
        self.box(LEFT,111,50,5,(.0,.62,.65))
        self.text('应急先行',LEFT,140,W-LEFT-RIGHT,38,True,color=BLUE)
        self.text('离线应急巡查 Agent',LEFT,205,W-LEFT-RIGHT,24,True)
        self.text('技术白皮书',LEFT,255,W-LEFT-RIGHT,30,True)
        self.text('原版 8080 · 可复现交付稿 v1.0',LEFT,320,W-LEFT-RIGHT,13,color=MUTED)
        self.box(LEFT,384,W-LEFT-RIGHT,130,(.93,.96,.985))
        self.text('部署方式 / 异构算力 / 真实性能 / 优化经验',LEFT+16,404,W-LEFT-RIGHT-32,12,True,color=BLUE)
        self.text('当前默认：CPU 推理\n可选 NPU：人员候选，历史验证尚未通过启用门槛\n证据：历史原始记录 + 本次 CPU 重测',LEFT+16,438,W-LEFT-RIGHT-32,11)
        self.text('2026 年 11 月 24 日',LEFT,640,W-LEFT-RIGHT,13,True)
        self.text('平台：Orange Pi 6 Plus / CIX P1 CD8180\n源码：emergency-agent-github\n原始项目及独立 NPU 实验均保留',LEFT,680,W-LEFT-RIGHT,10,color=MUTED)

    def toc(self,chapters,pages=None):
        self.chapter='文档导读';self.new_page()
        self.text('目录与阅读提示',LEFT,self.y,W-LEFT-RIGHT,23,True,color=BLUE);self.y+=56
        for chapter in chapters:
            label=chapter
            if pages and chapter in pages:label+=f'    {pages[chapter]}'
            self.text(label,LEFT,self.y,W-LEFT-RIGHT,11);self.y+=31
        self.y+=18
        self.paragraph('本文围绕原版实际代码和可追溯记录撰写。第 09 章按不同计时边界列出实测；第 10 章说明优化与量化失败经验；第 11–14 章提供复现、验证及来源。',10)
        self.paragraph('“部署成功”“软件测试通过”“模型精度合格”是不同结论。原版默认 CPU，NPU 数据引用历史板端结果；不将独立实验或模拟测试写成已上线能力。',10)

    def content(self,items):
        for kind,value in items:
            if kind=='chapter':
                self.chapter=value;self.new_page();self.chapters[value]=self.page
                self.y+=3
                h=self.text(value,LEFT,self.y,W-LEFT-RIGHT,21,True,color=BLUE)
                self.y+=h+24
            elif kind=='heading':
                self.ensure(75)
                h=self.text(value,LEFT,self.y,W-LEFT-RIGHT,13,True,color=BLUE);self.y+=h+12
            elif kind=='paragraph':self.paragraph(value)
            elif kind=='table':self.table(value)
            elif kind=='diagram':self.diagram()
            elif kind=='code':
                self.paragraph(value,8.4,mono=True)
                self.y+=3


def render(path,items,pages=None):
    surface=cairo.PDFSurface(str(path),W,H)
    surface.set_metadata(cairo.PDF_METADATA_TITLE,'应急先行：原版8080离线巡查Agent技术白皮书')
    surface.set_metadata(cairo.PDF_METADATA_SUBJECT,'部署、异构调用、实测与优化经验；2026-11-24')
    surface.set_metadata(cairo.PDF_METADATA_CREATE_DATE,'2026-11-24T17:00:00+08:00')
    book=Book(surface)
    book.cover();book.toc([v for k,v in items if k=='chapter'],pages);book.content(items)
    book.finish_page();surface.finish()
    return book.chapters,book.page


def main():
    items=list(tokens(SOURCE.read_text()))
    temporary=OUTPUT.with_suffix('.draft.pdf')
    pages,_=render(temporary,items)
    _,count=render(OUTPUT,items,pages)
    temporary.unlink()
    print(f'{OUTPUT} ({count} pages)')


if __name__=='__main__':main()
