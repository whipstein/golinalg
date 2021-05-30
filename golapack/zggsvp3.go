package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
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
func Zggsvp3(jobu, jobv, jobq byte, m, p, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, tola, tolb *float64, k, l *int, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, q *mat.CMatrix, ldq *int, iwork *[]int, rwork *mat.Vector, tau, work *mat.CVector, lwork, info *int) {
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
	lquery = ((*lwork) == -1)
	lwkopt = 1

	//     Test the input arguments
	(*info) = 0
	if !(wantu || jobu == 'N') {
		(*info) = -1
	} else if !(wantv || jobv == 'N') {
		(*info) = -2
	} else if !(wantq || jobq == 'N') {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*p) < 0 {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if (*lda) < maxint(1, *m) {
		(*info) = -8
	} else if (*ldb) < maxint(1, *p) {
		(*info) = -10
	} else if (*ldu) < 1 || (wantu && (*ldu) < (*m)) {
		(*info) = -16
	} else if (*ldv) < 1 || (wantv && (*ldv) < (*p)) {
		(*info) = -18
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -20
	} else if (*lwork) < 1 && !lquery {
		(*info) = -24
	}

	//     Compute workspace
	if (*info) == 0 {
		Zgeqp3(p, n, b, ldb, iwork, tau, work, toPtr(-1), rwork, info)
		lwkopt = int(work.GetRe(0))
		if wantv {
			lwkopt = maxint(lwkopt, *p)
		}
		lwkopt = maxint(lwkopt, minint(*n, *p))
		lwkopt = maxint(lwkopt, *m)
		if wantq {
			lwkopt = maxint(lwkopt, *n)
		}
		Zgeqp3(m, n, a, lda, iwork, tau, work, toPtr(-1), rwork, info)
		lwkopt = maxint(lwkopt, int(work.GetRe(0)))
		lwkopt = maxint(1, lwkopt)
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGSVP3"), -(*info))
		return
	}
	if lquery {
		return
	}

	//     QR with column pivoting of B: B*P = V*( S11 S12 )
	//                                           (  0   0  )
	for i = 1; i <= (*n); i++ {
		(*iwork)[i-1] = 0
	}
	Zgeqp3(p, n, b, ldb, iwork, tau, work, lwork, rwork, info)

	//     Update A := A*P
	Zlapmt(forwrd, m, n, a, lda, iwork)

	//     Determine the effective rank of matrix B.
	(*l) = 0
	for i = 1; i <= minint(*p, *n); i++ {
		if b.GetMag(i-1, i-1) > (*tolb) {
			(*l) = (*l) + 1
		}
	}

	if wantv {
		//        Copy the details of V, and form V.
		Zlaset('F', p, p, &czero, &czero, v, ldv)
		if (*p) > 1 {
			Zlacpy('L', toPtr((*p)-1), n, b.Off(1, 0), ldb, v.Off(1, 0), ldv)
		}
		Zung2r(p, p, toPtr(minint(*p, *n)), v, ldv, tau, work, info)
	}

	//     Clean up B
	for j = 1; j <= (*l)-1; j++ {
		for i = j + 1; i <= (*l); i++ {
			b.Set(i-1, j-1, czero)
		}
	}
	if (*p) > (*l) {
		Zlaset('F', toPtr((*p)-(*l)), n, &czero, &czero, b.Off((*l)+1-1, 0), ldb)
	}

	if wantq {
		//        Set Q = I and Update Q := Q*P
		Zlaset('F', n, n, &czero, &cone, q, ldq)
		Zlapmt(forwrd, n, n, q, ldq, iwork)
	}

	if (*p) >= (*l) && (*n) != (*l) {
		//        RQ factorization of ( S11 S12 ) = ( 0 S12 )*Z
		Zgerq2(l, n, b, ldb, tau, work, info)

		//        Update A := A*Z**H
		Zunmr2('R', 'C', m, n, l, b, ldb, tau, a, lda, work, info)
		if wantq {
			//           Update Q := Q*Z**H
			Zunmr2('R', 'C', n, n, l, b, ldb, tau, q, ldq, work, info)
		}
		//
		//        Clean up B
		//
		Zlaset('F', l, toPtr((*n)-(*l)), &czero, &czero, b, ldb)
		for j = (*n) - (*l) + 1; j <= (*n); j++ {
			for i = j - (*n) + (*l) + 1; i <= (*l); i++ {
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
	for i = 1; i <= (*n)-(*l); i++ {
		(*iwork)[i-1] = 0
	}
	Zgeqp3(m, toPtr((*n)-(*l)), a, lda, iwork, tau, work, lwork, rwork, info)

	//     Determine the effective rank of A11
	(*k) = 0
	for i = 1; i <= minint(*m, (*n)-(*l)); i++ {
		if a.GetMag(i-1, i-1) > (*tola) {
			(*k) = (*k) + 1
		}
	}

	//     Update A12 := U**H*A12, where A12 = A( 1:M, N-L+1:N )
	Zunm2r('L', 'C', m, l, toPtr(minint(*m, (*n)-(*l))), a, lda, tau, a.Off(0, (*n)-(*l)+1-1), lda, work, info)

	if wantu {
		//        Copy the details of U, and form U
		Zlaset('F', m, m, &czero, &czero, u, ldu)
		if (*m) > 1 {
			Zlacpy('L', toPtr((*m)-1), toPtr((*n)-(*l)), a.Off(1, 0), lda, u.Off(1, 0), ldu)
		}
		Zung2r(m, m, toPtr(minint(*m, (*n)-(*l))), u, ldu, tau, work, info)
	}

	if wantq {
		//        Update Q( 1:N, 1:N-L )  = Q( 1:N, 1:N-L )*P1
		Zlapmt(forwrd, n, toPtr((*n)-(*l)), q, ldq, iwork)
	}

	//     Clean up A: set the strictly lower triangular part of
	//     A(1:K, 1:K) = 0, and A( K+1:M, 1:N-L ) = 0.
	for j = 1; j <= (*k)-1; j++ {
		for i = j + 1; i <= (*k); i++ {
			a.Set(i-1, j-1, czero)
		}
	}
	if (*m) > (*k) {
		Zlaset('F', toPtr((*m)-(*k)), toPtr((*n)-(*l)), &czero, &czero, a.Off((*k)+1-1, 0), lda)
	}

	if (*n)-(*l) > (*k) {
		//        RQ factorization of ( T11 T12 ) = ( 0 T12 )*Z1
		Zgerq2(k, toPtr((*n)-(*l)), a, lda, tau, work, info)

		if wantq {
			//           Update Q( 1:N,1:N-L ) = Q( 1:N,1:N-L )*Z1**H
			Zunmr2('R', 'C', n, toPtr((*n)-(*l)), k, a, lda, tau, q, ldq, work, info)
		}

		//        Clean up A
		Zlaset('F', k, toPtr((*n)-(*l)-(*k)), &czero, &czero, a, lda)
		for j = (*n) - (*l) - (*k) + 1; j <= (*n)-(*l); j++ {
			for i = j - (*n) + (*l) + (*k) + 1; i <= (*k); i++ {
				a.Set(i-1, j-1, czero)
			}
		}

	}

	if (*m) > (*k) {
		//        QR factorization of A( K+1:M,N-L+1:N )
		Zgeqr2(toPtr((*m)-(*k)), l, a.Off((*k)+1-1, (*n)-(*l)+1-1), lda, tau, work, info)

		if wantu {
			//           Update U(:,K+1:M) := U(:,K+1:M)*U1
			Zunm2r('R', 'N', m, toPtr((*m)-(*k)), toPtr(minint((*m)-(*k), *l)), a.Off((*k)+1-1, (*n)-(*l)+1-1), lda, tau, u.Off(0, (*k)+1-1), ldu, work, info)
		}

		//        Clean up
		for j = (*n) - (*l) + 1; j <= (*n); j++ {
			for i = j - (*n) + (*k) + (*l) + 1; i <= (*m); i++ {
				a.Set(i-1, j-1, czero)
			}
		}

	}

	work.SetRe(0, float64(lwkopt))
}
