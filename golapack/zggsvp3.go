package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggsvp3 computes unitary matrices U, V and Q such that
//
//                    N-K-L  K    L
//  U**H*A*Q =     K ( 0    A12  A13 )  if M-K-L >= 0;
//                 L ( 0     0   A23 )
//             M-K-L ( 0     0    0  )
//
//                  N-K-L  K    L
//         =     K ( 0    A12  A13 )  if M-K-L < 0;
//             M-K ( 0     0   A23 )
//
//                  N-K-L  K    L
//  V**H*B*Q =   L ( 0     0   B13 )
//             P-L ( 0     0    0  )
//
// where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
// upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0,
// otherwise A23 is (M-K)-by-L upper trapezoidal.  K+L = the effective
// numerical rank of the (M+P)-by-N matrix (A**H,B**H)**H.
//
// This decomposition is the preprocessing step for computing the
// Generalized Singular Value Decomposition (GSVD), see subroutine
// ZGGSVD3.
func Zggsvp3(jobu, jobv, jobq byte, m, p, n int, a, b *mat.CMatrix, tola, tolb float64, u, v, q *mat.CMatrix, iwork *[]int, rwork *mat.Vector, tau, work *mat.CVector, lwork int) (k, l int, err error) {
	var forwrd, lquery, wantq, wantu, wantv bool
	var cone, czero complex128
	var i, j, lwkopt int

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	wantu = jobu == 'U'
	wantv = jobv == 'V'
	wantq = jobq == 'Q'
	forwrd = true
	lquery = (lwork == -1)
	lwkopt = 1

	//     Test the input arguments
	if !(wantu || jobu == 'N') {
		err = fmt.Errorf("!(wantu || jobu == 'N'): jobu='%c'", jobu)
	} else if !(wantv || jobv == 'N') {
		err = fmt.Errorf("!(wantv || jobv == 'N'): jobv='%c'", jobv)
	} else if !(wantq || jobq == 'N') {
		err = fmt.Errorf("!(wantq || jobq == 'N'): jobq='%c'", jobq)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 {
		err = fmt.Errorf("p < 0: p=%v", p)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, p) {
		err = fmt.Errorf("b.Rows < max(1, p): b.Rows=%v, p=%v", b.Rows, p)
	} else if u.Rows < 1 || (wantu && u.Rows < m) {
		err = fmt.Errorf("u.Rows < 1 || (wantu && u.Rows < m): jobu='%c', u.Rows=%v, m=%v", jobu, u.Rows, m)
	} else if v.Rows < 1 || (wantv && v.Rows < p) {
		err = fmt.Errorf("v.Rows < 1 || (wantv && v.Rows < p): jobv='%c', v.Rows=%v, p=%v", jobv, v.Rows, p)
	} else if q.Rows < 1 || (wantq && q.Rows < n) {
		err = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < n): jobq='%c', q.Rows=%v, n=%v", jobq, q.Rows, n)
	} else if lwork < 1 && !lquery {
		err = fmt.Errorf("lwork < 1 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
	}

	//     Compute workspace
	if err == nil {
		if err = Zgeqp3(p, n, b, iwork, tau, work, -1, rwork); err != nil {
			panic(err)
		}
		lwkopt = int(work.GetRe(0))
		if wantv {
			lwkopt = max(lwkopt, p)
		}
		lwkopt = max(lwkopt, min(n, p))
		lwkopt = max(lwkopt, m)
		if wantq {
			lwkopt = max(lwkopt, n)
		}
		if err = Zgeqp3(m, n, a, iwork, tau, work, -1, rwork); err != nil {
			panic(err)
		}
		lwkopt = max(lwkopt, int(work.GetRe(0)))
		lwkopt = max(1, lwkopt)
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zggsvp3", err)
		return
	}
	if lquery {
		return
	}

	//     QR with column pivoting of B: B*P = V*( S11 S12 )
	//                                           (  0   0  )
	for i = 1; i <= n; i++ {
		(*iwork)[i-1] = 0
	}
	if err = Zgeqp3(p, n, b, iwork, tau, work, lwork, rwork); err != nil {
		panic(err)
	}

	//     Update A := A*P
	Zlapmt(forwrd, m, n, a, iwork)

	//     Determine the effective rank of matrix B.
	l = 0
	for i = 1; i <= min(p, n); i++ {
		if b.GetMag(i-1, i-1) > tolb {
			l = l + 1
		}
	}

	if wantv {
		//        Copy the details of V, and form V.
		Zlaset(Full, p, p, czero, czero, v)
		if p > 1 {
			Zlacpy(Lower, p-1, n, b.Off(1, 0), v.Off(1, 0))
		}
		if err = Zung2r(p, p, min(p, n), v, tau, work); err != nil {
			panic(err)
		}
	}

	//     Clean up B
	for j = 1; j <= l-1; j++ {
		for i = j + 1; i <= l; i++ {
			b.Set(i-1, j-1, czero)
		}
	}
	if p > l {
		Zlaset(Full, p-l, n, czero, czero, b.Off(l, 0))
	}

	if wantq {
		//        Set Q = I and Update Q := Q*P
		Zlaset(Full, n, n, czero, cone, q)
		Zlapmt(forwrd, n, n, q, iwork)
	}

	if p >= l && n != l {
		//        RQ factorization of ( S11 S12 ) = ( 0 S12 )*Z
		if err = Zgerq2(l, n, b, tau, work); err != nil {
			panic(err)
		}

		//        Update A := A*Z**H
		if err = Zunmr2(Right, ConjTrans, m, n, l, b, tau, a, work); err != nil {
			panic(err)
		}
		if wantq {
			//           Update Q := Q*Z**H
			if err = Zunmr2(Right, ConjTrans, n, n, l, b, tau, q, work); err != nil {
				panic(err)
			}
		}
		//
		//        Clean up B
		//
		Zlaset(Full, l, n-l, czero, czero, b)
		for j = n - l + 1; j <= n; j++ {
			for i = j - n + l + 1; i <= l; i++ {
				b.Set(i-1, j-1, czero)
			}
		}

	}

	//     Let              N-L     L
	//                A = ( A11    A12 ) M,
	//
	//     then the following does the complete QR decomposition of A11:
	//
	//              A11 = U*(  0  T12 )*P1**H
	//                      (  0   0  )
	for i = 1; i <= n-l; i++ {
		(*iwork)[i-1] = 0
	}
	if err = Zgeqp3(m, n-l, a, iwork, tau, work, lwork, rwork); err != nil {
		panic(err)
	}

	//     Determine the effective rank of A11
	k = 0
	for i = 1; i <= min(m, n-l); i++ {
		if a.GetMag(i-1, i-1) > tola {
			k = k + 1
		}
	}

	//     Update A12 := U**H*A12, where A12 = A( 1:M, N-L+1:N )
	if err = Zunm2r(Left, ConjTrans, m, l, min(m, n-l), a, tau, a.Off(0, n-l), work); err != nil {
		panic(err)
	}

	if wantu {
		//        Copy the details of U, and form U
		Zlaset(Full, m, m, czero, czero, u)
		if m > 1 {
			Zlacpy(Lower, m-1, n-l, a.Off(1, 0), u.Off(1, 0))
		}
		if err = Zung2r(m, m, min(m, n-l), u, tau, work); err != nil {
			panic(err)
		}
	}

	if wantq {
		//        Update Q( 1:N, 1:N-L )  = Q( 1:N, 1:N-L )*P1
		Zlapmt(forwrd, n, n-l, q, iwork)
	}

	//     Clean up A: set the strictly lower triangular part of
	//     A(1:K, 1:K) = 0, and A( K+1:M, 1:N-L ) = 0.
	for j = 1; j <= k-1; j++ {
		for i = j + 1; i <= k; i++ {
			a.Set(i-1, j-1, czero)
		}
	}
	if m > k {
		Zlaset(Full, m-k, n-l, czero, czero, a.Off(k, 0))
	}

	if n-l > k {
		//        RQ factorization of ( T11 T12 ) = ( 0 T12 )*Z1
		if err = Zgerq2(k, n-l, a, tau, work); err != nil {
			panic(err)
		}

		if wantq {
			//           Update Q( 1:N,1:N-L ) = Q( 1:N,1:N-L )*Z1**H
			if err = Zunmr2(Right, ConjTrans, n, n-l, k, a, tau, q, work); err != nil {
				panic(err)
			}
		}

		//        Clean up A
		Zlaset(Full, k, n-l-k, czero, czero, a)
		for j = n - l - k + 1; j <= n-l; j++ {
			for i = j - n + l + k + 1; i <= k; i++ {
				a.Set(i-1, j-1, czero)
			}
		}

	}

	if m > k {
		//        QR factorization of A( K+1:M,N-L+1:N )
		if err = Zgeqr2(m-k, l, a.Off(k, n-l), tau, work); err != nil {
			panic(err)
		}

		if wantu {
			//           Update U(:,K+1:M) := U(:,K+1:M)*U1
			if err = Zunm2r(Right, NoTrans, m, m-k, min(m-k, l), a.Off(k, n-l), tau, u.Off(0, k), work); err != nil {
				panic(err)
			}
		}

		//        Clean up
		for j = n - l + 1; j <= n; j++ {
			for i = j - n + k + l + 1; i <= m; i++ {
				a.Set(i-1, j-1, czero)
			}
		}

	}

	work.SetRe(0, float64(lwkopt))

	return
}
