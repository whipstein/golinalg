package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgels solves overdetermined or underdetermined real linear systems
// involving an M-by-N matrix A, or its transpose, using a QR or LQ
// factorization of A.  It is assumed that A has full rank.
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
// 3. If TRANS = 'T' and m >= n:  find the minimum norm solution of
//    an underdetermined system A**T * X = B.
//
// 4. If TRANS = 'T' and m < n:  find the least squares solution of
//    an overdetermined system, i.e., solve the least squares problem
//                 minimize || B - A**T * X ||.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
func Dgels(trans mat.MatTrans, m, n, nrhs int, a, b *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var lquery, tpsd bool
	var anrm, bignum, bnrm, one, smlnum, zero float64
	var brow, i, iascl, ibscl, j, mn, nb, scllen, wsize int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments.
	mn = min(m, n)
	lquery = (lwork == -1)
	if !(trans == NoTrans || trans == Trans) {
		err = fmt.Errorf("!(trans == NoTrans || trans == Trans): trans=%s", trans)
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
		err = fmt.Errorf("lwork < max(1, mn+max(mn, nrhs)) && !lquery: lwork=%v, m=%v, n=%v, nrhs=%v, lquery=%v", lwork, m, n, nrhs, lquery)
		info = -10
	}

	//     Figure out optimal block size
	if info == 0 || info == -10 {

		tpsd = true
		if trans == NoTrans {
			tpsd = false
		}

		if m >= n {
			nb = Ilaenv(1, "Dgeqrf", []byte{' '}, m, n, -1, -1)
			if tpsd {
				nb = max(nb, Ilaenv(1, "Dormqr", []byte("LN"), m, nrhs, n, -1))
			} else {
				nb = max(nb, Ilaenv(1, "Dormqr", []byte("LT"), m, nrhs, n, -1))
			}
		} else {
			nb = Ilaenv(1, "Dgelqf", []byte{' '}, m, n, -1, -1)
			if tpsd {
				nb = max(nb, Ilaenv(1, "Dormlq", []byte("LT"), n, nrhs, m, -1))
			} else {
				nb = max(nb, Ilaenv(1, "Dormlq", []byte("LN"), n, nrhs, m, -1))
			}
		}

		wsize = max(1, mn+max(mn, nrhs)*nb)
		work.Set(0, float64(wsize))

	}

	if err != nil {
		gltest.Xerbla2("Dgels", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n, nrhs) == 0 {
		Dlaset(Full, max(m, n), nrhs, zero, zero, b)
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum) / Dlamch(Precision)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Scale A, B if max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', m, n, a, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Dlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Dlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Dlaset(Full, max(m, n), nrhs, zero, zero, b)
		goto label50
	}

	brow = m
	if tpsd {
		brow = n
	}
	bnrm = Dlange('M', brow, nrhs, b, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Dlascl('G', 0, 0, bnrm, smlnum, brow, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Dlascl('G', 0, 0, bnrm, bignum, brow, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 2
	}

	if m >= n {
		//        compute QR factorization of A
		if err = Dgeqrf(m, n, a, work, work.Off(mn), lwork-mn); err != nil {
			panic(err)
		}

		//        workspace at least N, optimally N*NB
		if !tpsd {
			//           Least-Squares Problem min || A * X - B ||
			//
			//           B(1:M,1:NRHS) := Q**T * B(1:M,1:NRHS)
			if err = Dormqr(Left, Trans, m, nrhs, n, a, work, b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
			if info, err = Dtrtrs(Upper, NoTrans, NonUnit, n, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			scllen = n

		} else {
			//           Underdetermined system of equations A**T * X = B
			//
			//           B(1:N,1:NRHS) := inv(R**T) * B(1:N,1:NRHS)
			if info, err = Dtrtrs(Upper, Trans, NonUnit, n, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			//           B(N+1:M,1:NRHS) = ZERO
			for j = 1; j <= nrhs; j++ {
				for i = n + 1; i <= m; i++ {
					b.Set(i-1, j-1, zero)
				}
			}

			//           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
			if err = Dormqr(Left, NoTrans, m, nrhs, n, a, work, b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = m

		}

	} else {
		//        Compute LQ factorization of A
		if err = Dgelqf(m, n, a, work, work.Off(mn), lwork-mn); err != nil {
			panic(err)
		}

		//        workspace at least M, optimally M*NB.
		if !tpsd {
			//           underdetermined system of equations A * X = B
			//
			//           B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS)
			if info, err = Dtrtrs(Lower, NoTrans, NonUnit, m, nrhs, a, b); err != nil {
				panic(err)
			}

			if info > 0 {
				return
			}

			//           B(M+1:N,1:NRHS) = 0
			for j = 1; j <= nrhs; j++ {
				for i = m + 1; i <= n; i++ {
					b.Set(i-1, j-1, zero)
				}
			}

			//           B(1:N,1:NRHS) := Q(1:N,:)**T * B(1:M,1:NRHS)
			if err = Dormlq(Left, Trans, n, nrhs, m, a, work, b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = n

		} else {
			//           overdetermined system min || A**T * X - B ||
			//
			//           B(1:N,1:NRHS) := Q * B(1:N,1:NRHS)
			if err = Dormlq(Left, NoTrans, n, nrhs, m, a, work, b, work.Off(mn), lwork-mn); err != nil {
				panic(err)
			}

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:M,1:NRHS) := inv(L**T) * B(1:M,1:NRHS)
			if info, err = Dtrtrs(Lower, Trans, NonUnit, m, nrhs, a, b); err != nil {
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
		if err = Dlascl('G', 0, 0, anrm, smlnum, scllen, nrhs, b); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Dlascl('G', 0, 0, anrm, bignum, scllen, nrhs, b); err != nil {
			panic(err)
		}
	}
	if ibscl == 1 {
		if err = Dlascl('G', 0, 0, smlnum, bnrm, scllen, nrhs, b); err != nil {
			panic(err)
		}
	} else if ibscl == 2 {
		if err = Dlascl('G', 0, 0, bignum, bnrm, scllen, nrhs, b); err != nil {
			panic(err)
		}
	}

label50:
	;
	work.Set(0, float64(wsize))

	return
}
