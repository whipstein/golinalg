package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgsja computes the generalized singular value decomposition (GSVD)
// of two complex upper triangular (or trapezoidal) matrices A and B.
//
// On entry, it is assumed that matrices A and B have the following
// forms, which may be obtained by the preprocessing subroutine ZGGSVP
// from a general M-by-N matrix A and P-by-N matrix B:
//
//              N-K-L  K    L
//    A =    K ( 0    A12  A13 ) if M-K-L >= 0;
//           L ( 0     0   A23 )
//       M-K-L ( 0     0    0  )
//
//            N-K-L  K    L
//    A =  K ( 0    A12  A13 ) if M-K-L < 0;
//       M-K ( 0     0   A23 )
//
//            N-K-L  K    L
//    B =  L ( 0     0   B13 )
//       P-L ( 0     0    0  )
//
// where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
// upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0,
// otherwise A23 is (M-K)-by-L upper trapezoidal.
//
// On exit,
//
//        U**H *A*Q = D1*( 0 R ),    V**H *B*Q = D2*( 0 R ),
//
// where U, V and Q are unitary matrices.
// R is a nonsingular upper triangular matrix, and D1
// and D2 are ``diagonal'' matrices, which are of the following
// structures:
//
// If M-K-L >= 0,
//
//                     K  L
//        D1 =     K ( I  0 )
//                 L ( 0  C )
//             M-K-L ( 0  0 )
//
//                    K  L
//        D2 = L   ( 0  S )
//             P-L ( 0  0 )
//
//                N-K-L  K    L
//   ( 0 R ) = K (  0   R11  R12 ) K
//             L (  0    0   R22 ) L
//
// where
//
//   C = diag( ALPHA(K+1), ... , ALPHA(K+L) ),
//   S = diag( BETA(K+1),  ... , BETA(K+L) ),
//   C**2 + S**2 = I.
//
//   R is stored in A(1:K+L,N-K-L+1:N) on exit.
//
// If M-K-L < 0,
//
//                K M-K K+L-M
//     D1 =   K ( I  0    0   )
//          M-K ( 0  C    0   )
//
//                  K M-K K+L-M
//     D2 =   M-K ( 0  S    0   )
//          K+L-M ( 0  0    I   )
//            P-L ( 0  0    0   )
//
//                N-K-L  K   M-K  K+L-M
// ( 0 R ) =    K ( 0    R11  R12  R13  )
//           M-K ( 0     0   R22  R23  )
//         K+L-M ( 0     0    0   R33  )
//
// where
// C = diag( ALPHA(K+1), ... , ALPHA(M) ),
// S = diag( BETA(K+1),  ... , BETA(M) ),
// C**2 + S**2 = I.
//
// R = ( R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N) and R33 is stored
//     (  0  R22 R23 )
// in B(M-K+1:L,N+M-K-L+1:N) on exit.
//
// The computation of the unitary transformation matrices U, V or Q
// is optional.  These matrices may either be formed explicitly, or they
// may be postmultiplied into input matrices U1, V1, or Q1.
func Ztgsja(jobu, jobv, jobq byte, m, p, n, k, l *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, tola, tolb *float64, alpha, beta *mat.Vector, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, q *mat.CMatrix, ldq *int, work *mat.CVector, ncycle, info *int) {
	var initq, initu, initv, upper, wantq, wantu, wantv bool
	var a2, b2, cone, czero, snq, snu, snv complex128
	var a1, a3, b1, b3, csq, csu, csv, _error, gamma, one, rwk, ssmin, zero float64
	var i, j, kcycle, maxit int

	maxit = 40
	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Decode and test the input parameters
	initu = jobu == 'I'
	wantu = initu || jobu == 'U'

	initv = jobv == 'I'
	wantv = initv || jobv == 'V'

	initq = jobq == 'I'
	wantq = initq || jobq == 'Q'

	(*info) = 0
	if !(initu || wantu || jobu == 'N') {
		(*info) = -1
	} else if !(initv || wantv || jobv == 'N') {
		(*info) = -2
	} else if !(initq || wantq || jobq == 'N') {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*p) < 0 {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if (*lda) < maxint(1, *m) {
		(*info) = -10
	} else if (*ldb) < maxint(1, *p) {
		(*info) = -12
	} else if (*ldu) < 1 || (wantu && (*ldu) < (*m)) {
		(*info) = -18
	} else if (*ldv) < 1 || (wantv && (*ldv) < (*p)) {
		(*info) = -20
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -22
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGSJA"), -(*info))
		return
	}

	//     Initialize U, V and Q, if necessary
	if initu {
		Zlaset('F', m, m, &czero, &cone, u, ldu)
	}
	if initv {
		Zlaset('F', p, p, &czero, &cone, v, ldv)
	}
	if initq {
		Zlaset('F', n, n, &czero, &cone, q, ldq)
	}

	//     Loop until convergence
	upper = false
	for kcycle = 1; kcycle <= maxit; kcycle++ {

		upper = !upper

		for i = 1; i <= (*l)-1; i++ {
			for j = i + 1; j <= (*l); j++ {

				a1 = zero
				a2 = czero
				a3 = zero
				if (*k)+i <= (*m) {
					a1 = a.GetRe((*k)+i-1, (*n)-(*l)+i-1)
				}
				if (*k)+j <= (*m) {
					a3 = a.GetRe((*k)+j-1, (*n)-(*l)+j-1)
				}

				b1 = b.GetRe(i-1, (*n)-(*l)+i-1)
				b3 = b.GetRe(j-1, (*n)-(*l)+j-1)

				if upper {
					if (*k)+i <= (*m) {
						a2 = a.Get((*k)+i-1, (*n)-(*l)+j-1)
					}
					b2 = b.Get(i-1, (*n)-(*l)+j-1)
				} else {
					if (*k)+j <= (*m) {
						a2 = a.Get((*k)+j-1, (*n)-(*l)+i-1)
					}
					b2 = b.Get(j-1, (*n)-(*l)+i-1)
				}

				Zlags2(upper, &a1, &a2, &a3, &b1, &b2, &b3, &csu, &snu, &csv, &snv, &csq, &snq)

				//              Update (K+I)-th and (K+J)-th rows of matrix A: U**H *A
				if (*k)+j <= (*m) {
					Zrot(l, a.CVector((*k)+j-1, (*n)-(*l)+1-1), lda, a.CVector((*k)+i-1, (*n)-(*l)+1-1), lda, &csu, toPtrc128(cmplx.Conj(snu)))
				}

				//              Update I-th and J-th rows of matrix B: V**H *B
				Zrot(l, b.CVector(j-1, (*n)-(*l)+1-1), ldb, b.CVector(i-1, (*n)-(*l)+1-1), ldb, &csv, toPtrc128(cmplx.Conj(snv)))

				//              Update (N-L+I)-th and (N-L+J)-th columns of matrices
				//              A and B: A*Q and B*Q
				Zrot(toPtr(minint((*k)+(*l), *m)), a.CVector(0, (*n)-(*l)+j-1), func() *int { y := 1; return &y }(), a.CVector(0, (*n)-(*l)+i-1), func() *int { y := 1; return &y }(), &csq, &snq)

				Zrot(l, b.CVector(0, (*n)-(*l)+j-1), func() *int { y := 1; return &y }(), b.CVector(0, (*n)-(*l)+i-1), func() *int { y := 1; return &y }(), &csq, &snq)

				if upper {
					if (*k)+i <= (*m) {
						a.Set((*k)+i-1, (*n)-(*l)+j-1, czero)
					}
					b.Set(i-1, (*n)-(*l)+j-1, czero)
				} else {
					if (*k)+j <= (*m) {
						a.Set((*k)+j-1, (*n)-(*l)+i-1, czero)
					}
					b.Set(j-1, (*n)-(*l)+i-1, czero)
				}

				//              Ensure that the diagonal elements of A and B are real.
				if (*k)+i <= (*m) {
					a.Set((*k)+i-1, (*n)-(*l)+i-1, a.GetReCmplx((*k)+i-1, (*n)-(*l)+i-1))
				}
				if (*k)+j <= (*m) {
					a.Set((*k)+j-1, (*n)-(*l)+j-1, a.GetReCmplx((*k)+j-1, (*n)-(*l)+j-1))
				}
				b.Set(i-1, (*n)-(*l)+i-1, b.GetReCmplx(i-1, (*n)-(*l)+i-1))
				b.Set(j-1, (*n)-(*l)+j-1, b.GetReCmplx(j-1, (*n)-(*l)+j-1))

				//              Update unitary matrices U, V, Q, if desired.
				if wantu && (*k)+j <= (*m) {
					Zrot(m, u.CVector(0, (*k)+j-1), func() *int { y := 1; return &y }(), u.CVector(0, (*k)+i-1), func() *int { y := 1; return &y }(), &csu, &snu)
				}

				if wantv {
					Zrot(p, v.CVector(0, j-1), func() *int { y := 1; return &y }(), v.CVector(0, i-1), func() *int { y := 1; return &y }(), &csv, &snv)
				}

				if wantq {
					Zrot(n, q.CVector(0, (*n)-(*l)+j-1), func() *int { y := 1; return &y }(), q.CVector(0, (*n)-(*l)+i-1), func() *int { y := 1; return &y }(), &csq, &snq)
				}

			}
		}

		if !upper {
			//           The matrices A13 and B13 were lower triangular at the start
			//           of the cycle, and are now upper triangular.
			//
			//           Convergence test: test the parallelism of the corresponding
			//           rows of A and B.
			_error = zero
			for i = 1; i <= minint(*l, (*m)-(*k)); i++ {
				goblas.Zcopy(toPtr((*l)-i+1), a.CVector((*k)+i-1, (*n)-(*l)+i-1), lda, work, func() *int { y := 1; return &y }())
				goblas.Zcopy(toPtr((*l)-i+1), b.CVector(i-1, (*n)-(*l)+i-1), ldb, work.Off((*l)+1-1), func() *int { y := 1; return &y }())
				Zlapll(toPtr((*l)-i+1), work, func() *int { y := 1; return &y }(), work.Off((*l)+1-1), func() *int { y := 1; return &y }(), &ssmin)
				_error = maxf64(_error, ssmin)
			}

			if math.Abs(_error) <= minf64(*tola, *tolb) {
				goto label50
			}
		}

		//        End of cycle loop
	}

	//     The algorithm has not converged after MAXIT cycles.
	(*info) = 1
	goto label100

label50:
	;

	//     If ERROR <= minint(TOLA,TOLB), then the algorithm has converged.
	//     Compute the generalized singular value pairs (ALPHA, BETA), and
	//     set the triangular matrix R to array A.
	for i = 1; i <= (*k); i++ {
		alpha.Set(i-1, one)
		beta.Set(i-1, zero)
	}

	for i = 1; i <= minint(*l, (*m)-(*k)); i++ {

		a1 = a.GetRe((*k)+i-1, (*n)-(*l)+i-1)
		b1 = b.GetRe(i-1, (*n)-(*l)+i-1)

		if a1 != zero {
			gamma = b1 / a1
			//
			if gamma < zero {
				goblas.Zdscal(toPtr((*l)-i+1), toPtrf64(-one), b.CVector(i-1, (*n)-(*l)+i-1), ldb)
				if wantv {
					goblas.Zdscal(p, toPtrf64(-one), v.CVector(0, i-1), func() *int { y := 1; return &y }())
				}
			}

			Dlartg(toPtrf64(math.Abs(gamma)), &one, beta.GetPtr((*k)+i-1), alpha.GetPtr((*k)+i-1), &rwk)

			if alpha.Get((*k)+i-1) >= beta.Get((*k)+i-1) {
				goblas.Zdscal(toPtr((*l)-i+1), toPtrf64(one/alpha.Get((*k)+i-1)), a.CVector((*k)+i-1, (*n)-(*l)+i-1), lda)
			} else {
				goblas.Zdscal(toPtr((*l)-i+1), toPtrf64(one/beta.Get((*k)+i-1)), b.CVector(i-1, (*n)-(*l)+i-1), ldb)
				goblas.Zcopy(toPtr((*l)-i+1), b.CVector(i-1, (*n)-(*l)+i-1), ldb, a.CVector((*k)+i-1, (*n)-(*l)+i-1), lda)
			}

		} else {

			alpha.Set((*k)+i-1, zero)
			beta.Set((*k)+i-1, one)
			goblas.Zcopy(toPtr((*l)-i+1), b.CVector(i-1, (*n)-(*l)+i-1), ldb, a.CVector((*k)+i-1, (*n)-(*l)+i-1), lda)
		}
	}

	//     Post-assignment
	for i = (*m) + 1; i <= (*k)+(*l); i++ {
		alpha.Set(i-1, zero)
		beta.Set(i-1, one)
	}

	if (*k)+(*l) < (*n) {
		for i = (*k) + (*l) + 1; i <= (*n); i++ {
			alpha.Set(i-1, zero)
			beta.Set(i-1, zero)
		}
	}

label100:
	;
	(*ncycle) = kcycle
}
