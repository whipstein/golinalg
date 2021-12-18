package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggglm solves a general Gauss-Markov linear model (GLM) problem:
//
//         minimize || y ||_2   subject to   d = A*x + B*y
//             x
//
// where A is an N-by-M matrix, B is an N-by-P matrix, and d is a
// given N-vector. It is assumed that M <= N <= M+P, and
//
//            rank(A) = M    and    rank( A B ) = N.
//
// Under these assumptions, the constrained equation is always
// consistent, and there is a unique solution x and a minimal 2-norm
// solution y, which is obtained using a generalized QR factorization
// of the matrices (A, B) given by
//
//    A = Q*(R),   B = Q*T*Z.
//          (0)
//
// In particular, if matrix B is square nonsingular, then the problem
// GLM is equivalent to the following weighted linear least squares
// problem
//
//              minimize || inv(B)*(d-A*x) ||_2
//                  x
//
// where inv(B) denotes the inverse of B.
func Zggglm(n, m, p int, a, b *mat.CMatrix, d, x, y, work *mat.CVector, lwork int) (info int, err error) {
	var lquery bool
	var cone, czero complex128
	var i, lopt, lwkmin, lwkopt, nb, nb1, nb2, nb3, nb4, np int

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	np = min(n, p)
	lquery = (lwork == -1)
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if m < 0 || m > n {
		err = fmt.Errorf("m < 0 || m > n: m=%v, n=%v", m, n)
	} else if p < 0 || p < n-m {
		err = fmt.Errorf("p < 0 || p < n-m: p=%v, m=%v, n=%v", p, m, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}

	//     Calculate workspace
	if err == nil {
		if n == 0 {
			lwkmin = 1
			lwkopt = 1
		} else {
			nb1 = Ilaenv(1, "Zgeqrf", []byte{' '}, n, m, -1, -1)
			nb2 = Ilaenv(1, "Zgerqf", []byte{' '}, n, m, -1, -1)
			nb3 = Ilaenv(1, "Zunmqr", []byte{' '}, n, m, p, -1)
			nb4 = Ilaenv(1, "Zunmrq", []byte{' '}, n, m, p, -1)
			nb = max(nb1, nb2, nb3, nb4)
			lwkmin = m + n + p
			lwkopt = m + np + max(n, p)*nb
		}
		work.SetRe(0, float64(lwkopt))

		if lwork < lwkmin && !lquery {
			err = fmt.Errorf("lwork < lwkmin && !lquery: lwork=%v, lwkmin=%v, lquery=%v", lwork, lwkmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zggglm", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Compute the GQR factorization of matrices A and B:
	//
	//          Q**H*A = ( R11 ) M,    Q**H*B*Z**H = ( T11   T12 ) M
	//                   (  0  ) N-M                 (  0    T22 ) N-M
	//                      M                         M+P-N  N-M
	//
	//     where R11 and T22 are upper triangular, and Q and Z are
	//     unitary.
	if err = Zggqrf(n, m, p, a, work, b, work.Off(m), work.Off(m+np), lwork-m-np); err != nil {
		panic(err)
	}
	lopt = int(work.GetRe(m + np + 1 - 1))

	//     Update left-hand-side vector d = Q**H*d = ( d1 ) M
	//                                               ( d2 ) N-M
	if err = Zunmqr(Left, ConjTrans, n, 1, m, a, work, d.CMatrix(max(1, n), opts), work.Off(m+np), lwork-m-np); err != nil {
		panic(err)
	}
	lopt = max(lopt, int(work.GetRe(m+np)))

	//     Solve T22*y2 = d2 for y2
	if n > m {
		if info, err = Ztrtrs(Upper, NoTrans, NonUnit, n-m, 1, b.Off(m, m+p-n), d.Off(m).CMatrix(n-m, opts)); err != nil {
			panic(err)
		}

		if info > 0 {
			info = 1
			return
		}

		y.Off(m+p-n).Copy(n-m, d.Off(m), 1, 1)
	}

	//     Set y1 = 0
	for i = 1; i <= m+p-n; i++ {
		y.Set(i-1, czero)
	}

	//     Update d1 = d1 - T12*y2
	if err = d.Gemv(NoTrans, m, n-m, -cone, b.Off(0, m+p-n), y.Off(m+p-n), 1, cone, 1); err != nil {
		panic(err)
	}

	//     Solve triangular system: R11*x = d1
	if m > 0 {
		if info, err = Ztrtrs(Upper, NoTrans, NonUnit, m, 1, a, d.CMatrix(m, opts)); err != nil {
			panic(err)
		}

		if info > 0 {
			info = 2
			return
		}

		//        Copy D to X
		x.Copy(m, d, 1, 1)
	}

	//     Backward transformation y = Z**H *y
	if err = Zunmrq(Left, ConjTrans, p, 1, np, b.Off(max(1, n-p+1)-1, 0), work.Off(m), y.CMatrix(max(1, p), opts), work.Off(m+np), lwork-m-np); err != nil {
		panic(err)
	}
	work.SetRe(0, float64(m+np+max(lopt, int(work.GetRe(m+np)))))

	return
}
