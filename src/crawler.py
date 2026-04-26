#gemini生成
import os
import time
import requests
from bs4 import BeautifulSoup

# 配置请求头（装作我们是正常用浏览器访问，防止被拦截）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def clean_text(text):
    """
    清理文本，去掉多余的空格、回车，只保留干净的正文。
    符合大模型训练前的脏数据处理要求！
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def get_neu_article(url):
    """抓取某一篇教务处或官网文章的正文"""
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = "utf-8"  # 学校官网绝大部分用 utf-8 或者 gbk

        if response.status_code != 200:
            print(f"❌ 页面抓取失败，状态码: {response.status_code}")
            return None

        # 开始解析网页
        soup = BeautifulSoup(response.text, "html.parser")

        # 【重点修改项】：这里要找到学校网站的正文用的是什么 HTML 标签。
        # 大部分学校网站正文会在类似 <div class="article_content"> 里面
        # 这里为了演示，我们先粗暴地拿走整个网页的文字：
        full_text = soup.get_text()

        return clean_text(full_text)

    except Exception as e:
        print(f"❌ 抓取出现错误: {e}")
        return None


def run_crawler():
    target_urls = [
        #这些不重要
    ]

    output_file = "scraped_data.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for index, url in enumerate(target_urls):
            print(f"⏳ 正在抓取第 {index+1}/{len(target_urls)} 篇...")

            article_text = get_neu_article(url)
            if article_text:
                f.write(article_text + "\n\n")
                print("✅ 抓取成功！")

            time.sleep(2)

    print(f"\n🎉 抓取完成！所有内容已保存至: {output_file}")


if __name__ == "__main__":
    run_crawler()