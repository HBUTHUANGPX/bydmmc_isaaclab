import sympy as sp
# ankle to AB
a, b, c, d, alpha, beta, phi, theta = sp.symbols('a b c d alpha beta phi theta')
# 对于 phi
# 变换后点D的坐标：


# $$\mathbf{D} = \begin{pmatrix} -a \cos \alpha + b \sin \beta \sin \alpha \\ -b \cos \beta \\ a \sin \alpha + b \sin \beta \cos \alpha \end{pmatrix}$$

D_x = -a * sp.cos(alpha) + b * sp.sin(beta) * sp.sin(alpha)
D_y = -b * sp.cos(beta)
D_z = a * sp.sin(alpha) + b * sp.sin(beta) * sp.cos(alpha)

#变换后点E的坐标：


# $$\mathbf{E} = \begin{pmatrix}-a \cos \alpha - b \sin \beta \sin \alpha \\ b \cos \beta \\ a \sin \alpha - b \sin \beta \cos \alpha \end{pmatrix}$$

E_x = -a * sp.cos(alpha) - b * sp.sin(beta) * sp.sin(alpha)
E_y = b * sp.cos(beta)
E_z = a * sp.sin(alpha) - b * sp.sin(beta) * sp.cos(alpha)

# C坐标为：

# $$\mathbf{C} = \begin{pmatrix}-a \cos \phi \\ -b \\ c+a \cos \phi \end{pmatrix}$$

C_x = -a * sp.cos(phi) 
C_y = -b
C_z = c + a * sp.cos(phi)


# $$\mathbf{F} = \begin{pmatrix}-a \cos \theta \\b\\ d+a\sin\theta \\  \end{pmatrix}$$

F_x = -a * sp.cos(theta)
F_y = b
F_z = d + a * sp.sin(theta)

#距离约束为$\|\mathbf{D} - \mathbf{C}\| = c$和$\|\mathbf{E} - \mathbf{F}\| = d$，即：
# $$(\mathbf{D}_x - \mathbf{C}_x)^2 + (\mathbf{D}_y - \mathbf{C}_y)^2 + (\mathbf{D}_z - \mathbf{C}_z)^2 = c^2$$
# $$(\mathbf{E}_x - \mathbf{F}_x)^2 + (\mathbf{E}_y - \mathbf{F}_y)^2 + (\mathbf{E}_z - \mathbf{F}_z)^2 = d^2$$
"""
约束$\| \mathbf{C} - \mathbf{D} \|^2 = c^2$展开后，可化为关于$k = a \cos \phi$的二次方程。定义辅助量：
$A = a \cos \alpha - b \sin \beta \sin \alpha,$
$B = c - a \sin \alpha - b \sin \beta \cos \alpha,$
$y^2 = b^2 (\cos \beta - 1)^2.$
二次方程为：
$2 k^2 + 2 (B - A) k + (A^2 + B^2 + y^2 - c^2) = 0.$
判别式：
$\Delta = [2 (B - A)]^2 - 8 (A^2 + B^2 + y^2 - c^2) = 4 (B - A)^2 - 8 (A^2 + B^2 + y^2 - c^2).$

若$\Delta \geq 0$，则：$k = \frac{ - (B - A) \pm \sqrt{ (B - A)^2 - 2 (A^2 + B^2 + y^2 - c^2) } }{2}.$
然后：

$\cos \phi = \frac{k}{a},$

且需满足$|\cos \phi| \leq 1$。若满足，则：

$\phi = \pm \arccos \left( \frac{k}{a} \right) + 2\pi n \quad (n \in \mathbb{Z}),$

具体取值取决于坐标系和角度范围（通常取主值$0 \leq \phi \leq \pi$)。可能存在两个解，对应$\pm$符号，选择物理上合理的那个。
"""
 # 对于 phi 的辅助量
A = a * sp.cos(alpha) - b * sp.sin(beta) * sp.sin(alpha)
B = c - a * sp.sin(alpha) - b * sp.sin(beta) * sp.cos(alpha)
y2 = b**2 * (sp.cos(beta) - 1)**2

# 二次方程系数
coeff_a = 2
coeff_b = 2 * (B - A)
coeff_c = A**2 + B**2 + y2 - c**2

# 判别式
Delta = coeff_b**2 - 4 * coeff_a * coeff_c

# k 解
k_plus = (-coeff_b + sp.sqrt(Delta)) / (2 * coeff_a)
k_minus = (-coeff_b - sp.sqrt(Delta)) / (2 * coeff_a) 

# phi 表达式
phi_plus = sp.acos(k_plus / a)
phi_minus = sp.acos(k_minus / a)

# 对于 theta 的辅助量
A_prime = a * sp.cos(alpha) + b * sp.sin(beta) * sp.sin(alpha)
B_prime = d - a * sp.sin(alpha) + b * sp.sin(beta) * sp.cos(alpha)

P = -2 * a * A_prime
Q = 2 * a * B_prime
K = d**2 - a**2 - A_prime**2 - B_prime**2 - y2

R = sp.sqrt(P**2 + Q**2)
gamma = sp.atan2(Q, P)

theta_plus = gamma + sp.acos(K / R)
theta_minus = gamma - sp.acos(K / R)

# 输出示例（符号形式）
print("phi possibilities:\r\n", phi_plus,"\r\n", phi_minus)
print("theta possibilities:\r\n", theta_plus,"\r\n", theta_minus)