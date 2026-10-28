"""17类售后意图标签定义（顺序即模型输出的类别id）"""

LABELS = [
    "物流查询",
    "退款申请",
    "退换货",
    "发票开具",
    "价格保护",
    "缺货补货",
    "商品质量问题",
    "错发漏发",
    "优惠券问题",
    "取消订单",
    "修改订单",
    "售后进度",
    "投诉举报",
    "会员积分",
    "安装维修",
    "其他咨询",
    "闲聊",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

NUM_LABELS = len(LABELS)
