import re
import os

endings = (
    #continue里deepseek整理的
    "。", "；", "：", ":", 
    "）", ")", 
    "……", "！", "!", 
    "？", "?", 
    "】", "}", "》", 
    "——"
)

def clean():
    with open(r"C:\Users\zjhsdld\Desktop\ai_to_learn\project1\src\data\neu_manual.txt","r",encoding="utf-8") as f:
        data=f.read()
    
    replacements = [
        #continue里deepseek读出的
        ("乐支", "东北"),
        ("乐止", "东北"),
        ("乐北", "东北"),
        ("乐苴", "东北"),
        ("乐龙", "东北"),
        ("乐花", "东北"),
        ("乐 K", "东北"),
        ("乐宜", "东北"),
        ("乐立", "东北"),
        ("东立", "东北"),
        ("东止", "东北"),
        ("东文", "东北"),
        ("东龙", "东北"),
        ("东：t", "东北"),
        ("东：«:", "东北"),
        ("东：H：", "东北"),
        ("东龙", "东北"),
        ("东文", "东北"),
        
        ("穴学", "大学"),
        ("大字", "大学"),
        ("太学", "大学"),
        
        ("学 生手册", "学生手册"),
        ("学生手 AB", "学生手册"),
        
        ("第w «翁管魏", "第四部分"),
        ("第爵赢 教I魏", "第三部分 教"),
        ("第翼毒 凝e魏", "第四部分 学"),
        ("第A雄 f 膏般", "第六部分 学"),
        ("第凝 fl 修 雄", "第五部分 团"),
        ("糊3乐 II: 穴字", "第七部分 创"),
        ("第-部分", "第一部分"),
        ("第=部分", "第二部分"),
        
        ("亥II 划", "刻画"),
        ("户 当", "户口应当"),
        ("户·当", "户口应当"),
        ("间隔I", "培训"),
        ("党I", "团委"),
        ("教I", "教务"),
        ("凝e", "学院"),
        ("凝B", "学院"),
        ("学凝", "学院"),
        ("f 膏", "学籍"),
        ("ii 修", "进修"),
        ("团lv", "团委"),
        ("的I", "的"),
        ("。I", "。"),
        ("。1", "。"),
        ("；I", "；"),
        ("：I", "："),
    ]
    
    for wrong, right in replacements:
        data=data.replace(wrong, right)
        
    data=re.sub(r'[-—]\s*\d+\s*[-—]','\n',data)
    
    data=re.sub(r'东北\s*大学\s*学生手册','\n',data)
    data=re.sub(r'[^\n]{0,8}[大穴][学字]\s*学生手册','\n',data)

    data = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', data) #去除字间空格，这句真牛逼，只能ai生成了以后也
    
    data = re.sub(r'(第[一二三四五六七八九十百]+)[·•・](章|节|条)', r'\1\2', data) #这谁能想到
    
    data=re.sub(r'(第[一二三四五六七八九十百]+章)',r'\n\1',data)
    data=re.sub(r'(第[一二三四五六七八九十百]+条)',r'\n\1',data)
    data=re.sub(r'(第[一二三四五六七八九十百]+节)',r'\n\1',data)
    
    data=re.sub(r'[\t]{1,}','',data) 
    data=re.sub(r'\n{2,}','\n\n',data)
    #去除多余空白
    
    ans=[]   
    last=""
        
    for line in data.splitlines():
        line=line.strip()
        if not line:
            if last:
                ans.append(last)
                last=""
            continue
        
        if not last:
            last=line
            
        elif line.startswith('东北大学'):
            last+="<|endoftext|>"
            ans.append(last)
            last=line
            
        elif last.endswith(endings) or (line.startswith('第')and ("部分" in line or "章" in line or "条" in line or "节" in line)):
            ans.append(last)
            last=line
            
        else:
            last+=line
            
    if last:
        ans.append(last)
        
    data='\n'.join(ans)
    
    with open(r"project1\src\data\output.txt","w",encoding="utf-8") as f:
        f.write(data)
    
    print('ok')
    
clean()
