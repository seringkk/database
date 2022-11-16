## 多路查找树（B树）



#### 背景

​		我们之前讨论的树，都是一个结点可以有多个孩子，但是结点自身只存储一个元素，二叉树限制更多，结点最多只能有两个孩子。一个结点只能存储一个元素，在元素非常多的时候，会使得要么树的度（结点拥有子树的个数的最大值）非常大，要么树的高度非常大，甚至两者都必须足够大才行，这显然造成了时间效率上的瓶颈，这迫使我们要打破每个结点只能存储一个元素的限制，因此引入了多路查找树的概念。

​		**多路查找树（muitl-way search tree），其每一个结点的孩子数可以多于两个，且每一个结点可以存储多个元素。**由于它是查找树，所有元素之间存在某种特定的排序关系。

​		在这里，每一个结点可以存储多少个元素，以及它的孩子数的多少是非常关键的。多路查找树有4种常见的形式：2-3树、2-3-4树、B树、B+树。

------



### 一、2-3树

#### 认识2-3树

​		**2-3树是这样的一棵多路查找树：其中的每一个结点都具有两个孩子（我们称它为2结点）或者三个孩子（称为3结点）**

​		**一个2结点包含一个元素和两个孩子（或者没有孩子 ）**，与二叉排序树类似，左孩子 <  根 < 右孩子。不同的是，这个2结点要么没有孩子，要么就有2个孩子（0 || 2）。

​		**一个3结点包含一大一小两个元素和三个孩子（或者没有孩子 ）**，这个3结点要么没有孩子，要么就有3个孩子（0 || 3）。如果某个3结点有孩子，左子树包含小于较小元素的元素，右子树包含大于较大元素的元素，中间子树包含介于两元素之间的元素。

​		2-3树的所有叶子都在同一层次上，如图8-8-2，是一棵有效2-3树。事实上，2-3树复杂的地方就在于结点的插入和删除，毕竟每个结点可能是2结点也可能是3结点，要保证所有叶子都在同一层次，是需要一番复杂操作的。



![1](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/1.png)



#### 拆解2-3树



- 2-3树的插入实现

​		2-3树的插入，与二叉排序树相同，插入操作一定是发生在叶子结点上的。不同的是，2-3树插入元素的过程可能会对该树的其余结点产生连锁反应。

插入可分为3种情况。

**①**对于空树，插入一个2结点即可，这很容易理解。



**②**插入结点到一个2结点的叶子上。由于2结点本身只有一个元素，所以将其升级为3结点即可。如同图8-8-3（图8-8-2的简化版），在左图插入元素3，遍历可知 3 < 4 < 8，于是只能考虑插入到叶子结点1的位置，自然想到将此结点变成一个3结点，像右图这样完成插入操作。

![2](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/2.png)



**③**往3结点中插入一个新元素。因为3结点本身是2-3树的结点最大容量（已经有两个元素），因此需要将其拆分，且将树中两元素或插入元素的三者中选择其一向上移动一层。复杂的情况正在于此。

**3-1.** 第一种情况 ，图8-8-4，需要在左图插入5。遍历可知 4 < 5 < 8，因此它需要插入在拥有6、7元素的3结点位置。问题就在于6、7已经是3结点，不能再升级。此时发现它的双亲结点4是个2结点，因此考虑让结点4升级为3结点，不过这样4结点就必须有3个孩子，于是想到将6、7拆分，让6与4结合成3结点，将5作为中间孩子，将7作为右孩子。

![3](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/3.png)



**3-2.** 另一种情况，图8-8-5，在左图插入元素11，遍历可得，9, 10 < 11 < 12, 14，因此它需要插入在拥有9、10元素的3结点位置。同3-1，发现9、10不能再增加结点，它们的双亲12、14也是3结点，也不能再插入元素。再往上看，12、14的双亲结点8是个2结点，于是想到将9、10拆分，12、14也拆分，让根结点8升级成3结点，最终变成右图所示。



![4](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/4.png)



**3-3.** 再看一个例子，图8-8-6，在左图插入元素2。经遍历可得 1 < 2 < 4，因此应该插入在拥有1、3元素的3结点位置，发现1、3和4、6都是3结点，不能再插入，甚至8、12也是3结点，那就意味着，当前的三层树结构已经不能满足当前的结点增加了。于是拆分1  3、4   6、8   2。最终变为右图所示。

![5](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/5.png)

通过这个例子，让我们发现，如果2-3树插入的传播效应导致根结点的拆分，则树的高度就会增加。



- 2-3树的删除实现

​		如果对前面的插入操作理解到位的话，删除操作应该不是难事。2-3树删除也分三种情况。与插入相反，我们从3结点开始讲起。

**①**所删元素位于3结点的叶子结点上。这很简单，只需要在该结点处删除该元素即可，不会影响到整棵树的其它结点结构。图8-8-7，删除元素9，只需将此结点改成只有元素10的2结点即可。

![6](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/6.png)



---



**②**所删元素位于2结点上，即要删除的是只有一个元素的结点。2-3树不同于普通的树结构，直接删除是不行的，图8-8-8，如果删除了结点1，那么4本来是个2结点（拥有两个孩子），此时就不满足定义了。

![7](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/7.png)

因此，对于删除叶子是2结点的情况，需要分为4种形势来处理。

**2-1.** 此结点的双亲也是2结点，且拥有一个3结点的右孩子。图8-8-9，这时删除结点1，那么只需要左旋，即6成为双亲，4成为左孩子，7成为右孩子。



![8](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/8.png)

**2-2.** 此结点双亲是2结点，它的右孩子也是2结点。图8-8-10，此时删除结点4，如果直接左旋会造成没有右孩子，因此需要对整棵树变形。做法就是，我们目标是让结点7变成3结点，那就得让比7稍大的元素8下来，随即就得让比元素8稍大的元素9补充结点8的位置，于是就有了8-8-10的中间的图，于是再右旋，变成右图结果。



![9](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/9.png)

**2-3.** 此结点双亲是个3结点。图8-8-11，此时删除结点10，意味着双亲12、14这个结点不再是3结点（删除10，双亲不再拥有3个孩子），于是将此结点拆分，并将12与13合并成为左孩子。



![10](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/10.png)

**2-4.** 如果当前树是一个满二叉树的情况，此时删除任何一个叶子都会使得整棵树不再满足2-3树的定义。图8-8-12，假设删除叶子结点8时，就不得不考虑将2-3树的层数减少，办法是将8的双亲和其左子树6合并成为一个3结点，再将9和14合并成3结点，结果如右图。



![11](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/11.png)

---



**③**所删除的元素位于非叶子的分支结点。此时我们通常是将树按中序遍历后得到此元素的前驱或后继元素，考虑让它们来补位即可。

**3-1.** 如果要删除的分支结点是2结点。图8-8-13，删除结点4，分析得到它的前驱是1后继是6，显然，6、7是3结点，只需要用6来补位即可，如右图。



![12](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/12.png)

**3-2.**如果要删除的分支是3结点的某一元素，图8-8-14，删除12、14结点的12，简单分析，删除12后将左孩子10上升到删除位置合适。



![13](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/13.png)

​		最后，如果对2-3树的插入删除等所有情况都进行讲解，既占篇幅，有没必要，这些操作都是有规律的，需要自己在实践中体会。



### 二、2-3-4树

​		2-3-4树其实就是2-3树的概念拓展，包括了4结点的使用。**一个4结点包含小、中、大三个元素和四个孩子（或者没有孩子）**如果某4结点有孩子的话，左子树包含小于最小元素的元素，第二子树包含大于最小元素，小于第二元素的元素；第三子树包含大于第二元素，小于最大元素的元素；右子树包含大于最大元素的元素。



​		由于2-3-4树和2-3树是类似的，这里就简单演示一下，构建一个数组为{7，1，2，5，6，9，8，4，3}的2-3-4树的过程，图8-8-15。图1是分别插入1、2、7的结果，因为这时3个元素满足2-3-4树单个4结点的定义，而插入元素5后，超过了4结点的定义，因此拆分成图2。之后的图都是在元素不断插入最后形成图7的2-3-4树。

![14](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/14.png)



图8-8-16是对一个2-3-4树的删除结点的演变过程，删除顺序是1、6、3、4、5、2、9。

![15](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/15.png)



### 三、B树

​		B树是多路查找树的主角，现在才出现似乎太晚了，其实，前面一直都在将B树。

​		**B树是一种平衡的多路查找树**，2-3树和2-3-4树都是B树的特例。**结点最大的孩子数目称为B树的阶（order）**，因此，2-3树是3阶B树，2-3-4树是4阶B树。

一个m阶B树具有如下属性 :

![16](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/16.png)



​		例如，在讲2-3-4树时插入9个数后的图转成B树示意就如图8-8-17的右图所示，左侧灰色方块表示当前结点的元素个数。（**A2**所指的结点元素6、7大于**A1**和**A0**所指元素4、1、2；**A3**所指元素9，大于**K3**元素8）



![17](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/17.png)

​		在B树上查找的过程是一个顺指针查找结点，和在结点中查找关键字的交叉过程。例如查找元素7，首先从外存（如硬盘）读取到根结点的3、5、8三个元素，发现7不在其中，但是在5、8之间，因此就通过A2再读取外存的6、7结点，查找到所要的元素。至于B树的删除，方式与2-3树、2-3-4树类似，只不过阶数可能会很大而已。



![18](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/18.png)



### 四、B+树

​		尽管前面讲了B树的诸多好处，但其实它还是有缺陷的，对于树结构来说，我们都可以通过中序遍历来顺序查找树中的元素，这一切都是在内存中运行的。

![19](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/19.png)

![20](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/20.png)



![21](https://github.com/kyrian330/Data-Structure-Algorithm/blob/main/%E7%AE%97%E6%B3%95/%E6%9F%A5%E6%89%BE/img/%E9%AB%98%E7%BA%A7%E6%9F%A5%E6%89%BE/21.png)

上述图片摘抄与《大话数据结构》，更多详细内容可以参考《算法导论》



### 拆解B树

代码太复杂，有时间再拆解，有能力的自己看完整代码。

- 完整代码

```c
#include <stdio.h>
#include <stdlib.h>

#define m 3 // B树的阶，暂设为3
#define N 17 // 数据元素个数

typedef struct BTNode {
    int keynum;   // 结点中关键字个数，即结点的大小
    struct BTNode *parent;  // 指向双亲结点

    struct Node {   // 结点向量类型
        int key;   // 关键字向量
        struct BTNode *ptr;   // 子树指针向量
        int recptr;   // 记录指针向量
    }node[m+1];   // key,recptr的0号单元未用
}BTNode,*BTree;   // B树结点和B树的类型

typedef struct {
    BTNode *pt;  // 指向找到的结点
    int i;   // 1..m，在结点中的关键字序号
    int tag;   // 1:查找成功，O:查找失败
}Result;   // B树的查找结果类型

// 在p->node[1..keynum].key中查找i,使得p->node[i].key≤K＜p->node[i+1].key
int Search(BTNode* p, int K) {
    int i=0,j;
    for(j=1;j<=p->keynum;j++)
        if(p->node[j].key<=K)
            i=j;
    return i;
}

// 在m阶B树T上查找关键字K，返回结果(pt,i,tag)。若查找成功，则特征值
// tag=1，指针pt所指结点中第i个关键字等于K；否则特征值tag=0，等于K的
// 关键字应插入在指针Pt所指结点中第i和第i+1个关键字之间。
Result SearchBTree(BTNode* T, int K) {
    BTNode* p=T,*q=NULL;   // 初始化，p指向待查结点，q指向p的双亲
    int found=0;
    int i=0;
    Result r;
    while(p&&!found) {
        i=Search(p,K);   // p->node[i].key≤K<p->node[i+1].key
        if(i>0&&p->node[i].key==K)   // 找到待查关键字
            found=1;
        else {
            q=p;
            p=p->node[i].ptr;
        }
    }
    r.i=i;
    if(found) {   // 查找成功
        r.pt=p;
        r.tag=1;
    }
    else {  // 查找不成功，返回K的插入位置信息
        r.pt=q;
        r.tag=0;
    }
    return r;
}

//将r->key、r和ap分别插入到q->key[i+1]、q->recptr[i+1]和q->ptr[i+1]中
void Insert(BTNode*& q,int i,int key,BTNode* ap) {
    int j;
    for(j=q->keynum;j>i;j--)   // 空出q->node[i+1]
        q->node[j+1]=q->node[j];
    q->node[i+1].key=key;
    q->node[i+1].ptr=ap;
    q->node[i+1].recptr=key;
    q->keynum++;
}

//将结点q分裂成两个结点，前一半保留，后一半移入新生结点ap
void split(BTNode*& q, BTNode*& ap) {
    int i,s=(m+1)/2;
    ap = (BTNode*)malloc(sizeof(BTNode));   // 生成新结点ap
    ap->node[0].ptr=q->node[s].ptr;   // 后一半移入ap
    for(i=s+1;i<=m;i++) {
        ap->node[i-s]=q->node[i];
        if(ap->node[i-s].ptr)
            ap->node[i-s].ptr->parent=ap;
    }
    ap->keynum=m-s;
    ap->parent=q->parent;
    q->keynum=s-1;   // q的前一半保留，修改keynum
}

//生成含信息(T,r,ap)的新的根结点&T，原T和ap为子树指针
void NewRoot(BTNode*&T,int key,BTNode* ap) {
    BTNode* p;
    p=(BTNode*)malloc(sizeof(BTNode));
    p->node[0].ptr=T;
    T=p;
    if(T->node[0].ptr)
        T->node[0].ptr->parent=T;
    T->parent=NULL;
    T->keynum=1;
    T->node[1].key=key;
    T->node[1].recptr=key;
    T->node[1].ptr=ap;
    if(T->node[1].ptr)
        T->node[1].ptr->parent=T;
}

// 在m阶B树T上结点*q的key[i]与key[i+1]之间插入关键字K的指针r。若引起
// 结点过大,则沿双亲链进行必要的结点分裂调整,使T仍是m阶B树。
void InsertBTree(BTNode*&T,int key,BTNode* q,int i) {
    BTNode* ap=NULL;
    int finished=0;
    int s;
    int rx;
    rx=key;
    while(q&&!finished) {
        Insert(q,i,rx,ap);   // 将r->key、r和ap分别插入到q->key[i+1]、q->recptr[i+1]和q->ptr[i+1]中
        if(q->keynum<m)
            finished=1;   // 插入完成
        else {
            // 分裂结点*q
            s=(m+1)/2;
            rx=q->node[s].recptr;
            split(q,ap); // 将q->key[s+1..m],q->ptr[s..m]和q->recptr[s+1..m]移入新结点*ap
            q=q->parent;
            if(q)
                i=Search(q,key); // 在双亲结点*q中查找rx->key的插入位置
        }
    }
    if(!finished) // T是空树(参数q初值为NULL)或根结点已分裂为结点*q和*ap
        NewRoot(T,rx,ap); // 生成含信息(T,rx,ap)的新的根结点*T，原T和ap为子树指针
}


void print(BTNode c,int i) { // TraverseDSTable()调用的函数
    printf("(%d)",c.node[i].key);
}



int main()
{
    int r[N]={22,16,41,58,8,11,12,16,17,22,23,31,41,52,58,59,61};
//    int r[N]={7,1,2,5,6,9,8,4,3};
    BTNode* T=NULL;
    Result s;
    int i;
    for(i=0;i<N;i++) {
        s=SearchBTree(T,r[i]);
        if(!s.tag)
            InsertBTree(T,r[i],s.pt,s.i);
    }
    printf("请输入待查找记录的关键字:");
    scanf("%d",&i);
    s=SearchBTree(T,i);
    if(s.tag)
        print(*(s.pt),s.i);
    else
        printf("没找到");
    printf("\n");
    return 0;
}
```

