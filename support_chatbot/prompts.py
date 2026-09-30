# prompts.py
# 11-12 hafta: Prompt Versioning
#
# G'oya: system promptlarni kod ichida "qattiq" yozish o'rniga, har birini VERSIYALAB
# saqlaymiz. Har bir so'rov logida QAYSI versiya ishlatilgani yoziladi — shuning uchun
# keyinchalik "v2 javoblari nega yomonlashdi?" kabi savollarga aniq javob topish mumkin.

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    text: str
    notes: str = ""


# Har bir nom uchun versiyalar tarixi (eskisidan yangisiga). Oxirgisi = "latest".
_REGISTRY = {
    "support_chatbot": [
        PromptVersion(
            name="support_chatbot",
            version="v1",
            text=(
                "Siz {company_name} kompaniyasining qo'llab-quvvatlash botisiz. "
                "Foydalanuvchi savollariga kontekst asosida javob bering."
            ),
            notes="Boshlang'ich versiya",
        ),
        PromptVersion(
            name="support_chatbot",
            version="v2",
            text=(
                "Siz {company_name} kompaniyasining qo'llab-quvvatlash botisiz.\n\n"
                "QOIDALAR:\n"
                "1. FAQAT berilgan kontekstga asoslaning.\n"
                "2. Agar javob kontekstda yo'q bo'lsa: \"Bu ma'lumot hujjatlarda topilmadi\" deng.\n"
                "3. Har bir faktdan keyin manba faylini ko'rsating.\n"
                "4. Qisqa va aniq bo'ling."
            ),
            notes="v1'ga hallucination guard va source citation qo'shildi (7-8 haftadagi tajriba asosida)",
        ),
    ],
}


def get_prompt(name, version="latest", **format_kwargs):
    """
    Promptni nomi va versiyasi bo'yicha qaytaradi.
    version="latest" -> ro'yxatdagi oxirgi (eng yangi) versiya.
    format_kwargs -> promptdagi {placeholder}larni to'ldirish uchun.

    Qaytaradi: (matn, ishlatilgan_versiya_raqami)
    """
    history = _REGISTRY.get(name)
    if not history:
        raise KeyError(f"'{name}' nomli prompt topilmadi. Mavjudlar: {list(_REGISTRY)}")

    if version == "latest":
        chosen = history[-1]
    else:
        matches = [p for p in history if p.version == version]
        if not matches:
            available = [p.version for p in history]
            raise KeyError(f"'{name}' uchun '{version}' versiyasi topilmadi. Mavjud: {available}")
        chosen = matches[0]

    text = chosen.text.format(**format_kwargs) if format_kwargs else chosen.text
    return text, chosen.version


def list_versions(name):
    return [(p.version, p.notes) for p in _REGISTRY.get(name, [])]