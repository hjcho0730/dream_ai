#region import *
from tkinter import *
from modelLoad import modelLoad, preparePre, file_name
from func.utils import *
from func.main_func import process_data, res
from tkinter import ttk
#endregion

root= Tk()

#region 메인 기능
cnt= 0
fileName= file_name()
path = os.path.join(real_path, "models", fileName)
loadedModel= modelLoad(path=path)
getVec_steps= preparePre()
def cmd():
    global cnt
    s= entry.get()

    new_vec= process_data([s], getVec_steps)
    new_vec = toArray(new_vec)

    predictions = loadedModel.predict(new_vec)
    label =  res[int(predictions[0])]
    print(f'"{s}" → {label}')

    result= f"결과: {label}"
    resTxt.config(text=result)

    result= f"{label}: {s}"
    cnt+= 1
    size= cnt
    treeview.insert('', 'end', text=str(size), values=(label, s), iid=str(size)+"번")
    treeview.yview_moveto(1.0)
#endregion
#region 창 설정
root.title('꿈나비 댓글 분류기')
root.geometry("700x400+100+100")
root.resizable(False, False)
#endregion
#region 위젯 생성
txt1= Label(root, text= "댓글 분류기")
entry= Entry(root, width=60)
btn1= Button(root, width= 16, text= "입력", command= cmd)
resTxt= Label(root, text= "(여기에 결과가 출력됩니다)")
frame= Frame(root, height= 40)

treeview=ttk.Treeview(frame, columns=["one", "two"], displaycolumns=["two", "one"])
#endregion
#region 트리뷰 헤드
treeview.column("#0", width=70)
treeview.heading("#0", text="번호")
treeview.column("one", width=100, anchor="w")
treeview.heading("one", text="결과", anchor="center")
treeview.column("#2", width=100, anchor="center")
treeview.heading("two", text="문장", anchor="center")
#endregion
#region 스크롤 설정
sb= Scrollbar(frame, orient="vertical")
sb.config(command=treeview.yview)

treeview.config(yscrollcommand=sb.set)
#endregion
#region 위젯 팩
txt1.pack()
entry.pack()
btn1.pack()
resTxt.pack()
frame.pack()
treeview.pack(side="left")
sb.pack(side="right", fill="y")
#endregion

root.mainloop()
