package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgels solves overdetermined or underdetermined complex linear systems
// involving an M-by-N matrix A, or its conjugate-transpose, using a QR
// or LQ factorization of A.  It is assumed that A has full rank.
//
// The following options are provided:
//
// 1. If TRANS = 'N' and m >= n:  find the least squares solution of
//    an overdetermined system, i.e., solve the least squares problem
//                 minimize || B - A*X ||.
//
// 2. If TRANS = 'N' and m < n:  find the minimum norm solution of
//    an underdetermined system A * X = B.
//
// 3. If TRANS = 'C' and m >= n:  find the minimum norm solution of
//    an underdetermined system A**H * X = B.
//
// 4. If TRANS = 'C' and m < n:  find the least squares solution of
//    an overdetermined system, i.e., solve the least squares problem
//                 minimize || B - A**H * X ||.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
func Zgels(trans mat.MatTrans, m, n, nrhs int, a, b *mat.CMatrix, work *mat.CVector, lwork int) (info int, err error) {
	var lquery, tpsd bool
	var czero complex128
	var anrm, bignum, bnrm, one, smlnum, zero float64
	var brow, i, iascl, ibscl, j, mn, nb, scllen, wsize int

	rwork := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)

	//     Test the input arguments.
	mn = min(m, n)
	lquery = (lwork == -1)
	if !(trans == NoTrans || trans == ConjTrans) {
		err = fmt.Errorf("!(trans == NoTrans || trans == ConjTrans): trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, m, n) {
		err = fmt.Errorf("b.Rows < max(1, m, n): b.Rows=%v, m=%v, n=%v", b.Rows, m, n)
	} else if lwork < max(1, mn+max(mn, nrhs)) && !lquery {
		err = fmt.Errorf("lwork < max(1, mn+max(mn, nrhs)) && !lquery: lwork=%v, mn=%v, nrhs=%v, lquery=%v", lwork, mn, nrhs, lquery)
	}

	//     Figure out optimal block size
	if err == nil || (lwork < max(1, mn+max(mn, nrhs)) && !lquery) {

		tpsd = true
		if trans == NoTrans {
			tpsd = false
		}

		if m >= n {
			nb = Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, -1, -1)
			if tpsd {
				nb = max(nb, Ilaenv(1, "Zunmqr", []byte("LN"), m, nrhs, n, -1))
			} else {
				nb = max(nb, Ilaenv(1, "Zunmqr", []byte("LC"), m, nrhs, n, -1))
			}
		} else {
			nb = Ilaenv(1, "Zgelqf", []byte{' '}, m, n, -1, -1)
			if tpsd {
				nb = max(nb, Ilaenv(1, "Zunmlq", []byte("LC"), n, nrhs, m, -1))
			} else {
				nb = max(nb, Ilaenv(1, "Zunmlq", []byte("LN"), n, nrhs, m, -1))
			}
		}

		wsize = max(1, mn+max(mn, nrhs)*nb)
		work.SetRe(0, float64(wsize))

	}

	if err != nil {
		gltest.Xerbla2("Zgels", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n, nrhs) == 0 {
		Zlaset(Full, max(m, n), nrhs, czero, czero, b)
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum) / Dlamch(Precision)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Scale A, B if max element outside range [SMLNUM,BIGNUM]
	//
	anrm = Zlange('M', m, n, a, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Zlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Zlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Zlaset(Full, max(m, n), nrhs, czero, czero, b)
		goto label50
	}
	//
	brow = m
	if tpsd {
		brow = n
	}
	bnrm = Zlange('M', brow, nrhs, b, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Zlascl('G', 0, 0, bnrm, smlnum, brow, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Zlascl('G', 0, 0, bnrm, bignum, brow, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 2
	}

	if m >= n {
		//        compute QR factorization of A
		if err = Zgeqrf(m, n, a, work.Off(0), work.Off(mn), lwork-mn); err != nil {
			panic(err)
		}

		//        workspace at least N, optimally N*NB
		if !tpsd {
			//           Least-Squares Problem min || A * X - B ||
			//
			//           B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS)
			if err = Zunmqr(Left, ConjTrans, m, nrhs, n, a, work.Off(0), b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
			if info, err = Ztrtrs(Upper, NoTrans, NonUnit, n, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			scllen = n

		} else {
			//           Underdetermined system of equations A**T * X = B
			//
			//           B(1:N,1:NRHS) := inv(R**H) * B(1:N,1:NRHS)
			if info, err = Ztrtrs(Upper, ConjTrans, NonUnit, n, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			//           B(N+1:M,1:NRHS) = ZERO
			for j = 1; j <= nrhs; j++ {
				for i = n + 1; i <= m; i++ {
					b.Set(i-1, j-1, czero)
				}
			}

			//           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
			if err = Zunmqr(Left, NoTrans, m, nrhs, n, a, work.Off(0), b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = m

		}

	} else {
		//        Compute LQ factorization of A
		if err = Zgelqf(m, n, a, work.Off(0), work.Off(mn), lwork-mn); err != nil {
			panic(err)
		}

		//        workspace at least M, optimally M*NB.
		if !tpsd {
			//           underdetermined system of equations A * X = B
			//
			//           B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS)
			if info, err = Ztrtrs(Lower, NoTrans, NonUnit, m, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			//           B(M+1:N,1:NRHS) = 0
			for j = 1; j <= nrhs; j++ {
				for i = m + 1; i <= n; i++ {
					b.Set(i-1, j-1, czero)
				}
			}

			//           B(1:N,1:NRHS) := Q(1:N,:)**H * B(1:M,1:NRHS)
			if err = Zunmlq(Left, ConjTrans, n, nrhs, m, a, work.Off(0), b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = n

		} else {
			//           overdetermined system min || A**H * X - B ||
			//
			//           B(1:N,1:NRHS) := Q * B(1:N,1:NRHS)
			if err = Zunmlq(Left, NoTrans, n, nrhs, m, a, work.Off(0), b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:M,1:NRHS) := inv(L**H) * B(1:M,1:NRHS)
			if info, err = Ztrtrs(Lower, ConjTrans, NonUnit, m, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			scllen = m

		}

	}

	//     Undo scaling
	if iascl == 1 {
		if err = Zlascl('G', 0, 0, anrm, smlnum, scllen, nrhs, b); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Zlascl('G', 0, 0, anrm, bignum, scllen, nrhs, b); err != nil {
			panic(err)
		}
	}
	if ibscl == 1 {
		if err = Zlascl('G', 0, 0, smlnum, bnrm, scllen, nrhs, b); err != nil {
			panic(err)
		}
	} else if ibscl == 2 {
		if err = Zlascl('G', 0, 0, bignum, bnrm, scllen, nrhs, b); err != nil {
			panic(err)
		}
	}

label50:
	;
	work.SetRe(0, float64(wsize))

	return
}
