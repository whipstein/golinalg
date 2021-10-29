package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelsy computes the minimum-norm solution to a complex linear least
// squares problem:
//     minimize || A * X - B ||
// using a complete orthogonal factorization of A.  A is an M-by-N
// matrix which may be rank-deficient.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
//
// The routine first computes a QR factorization with column pivoting:
//     A * P = Q * [ R11 R12 ]
//                 [  0  R22 ]
// with R11 defined as the largest leading submatrix whose estimated
// condition number is less than 1/RCOND.  The order of R11, RANK,
// is the effective rank of A.
//
// Then, R22 is considered to be negligible, and R12 is annihilated
// by unitary transformations from the right, arriving at the
// complete orthogonal factorization:
//    A * P = Q * [ T11 0 ] * Z
//                [  0  0 ]
// The minimum-norm solution is then
//    X = P * Z**H [ inv(T11)*Q1**H*B ]
//                 [        0         ]
// where Q1 consists of the first RANK columns of Q.
//
// This routine is basically identical to the original xGELSX except
// three differences:
//   o The permutation of matrix B (the right hand side) is faster and
//     more simple.
//   o The call to the subroutine xGEQPF has been substituted by the
//     the call to the subroutine xGEQP3. This subroutine is a Blas-3
//     version of the QR factorization with column pivoting.
//   o Matrix B (the right hand side) is updated with Blas-3.
func Zgelsy(m, n, nrhs int, a, b *mat.CMatrix, jpvt *[]int, rcond float64, work *mat.CVector, lwork int, rwork *mat.Vector) (rank int, err error) {
	var lquery bool
	var c1, c2, cone, czero, s1, s2 complex128
	var anrm, bignum, bnrm, one, smax, smaxpr, smin, sminpr, smlnum, wsize, zero float64
	var i, iascl, ibscl, imax, imin, ismax, ismin, j, lwkopt, mn, nb, nb1, nb2, nb3, nb4 int

	imax = 1
	imin = 2
	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	mn = min(m, n)
	ismin = mn + 1
	ismax = 2*mn + 1

	//     Test the input arguments.
	nb1 = Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, -1, -1)
	nb2 = Ilaenv(1, "Zgerqf", []byte{' '}, m, n, -1, -1)
	nb3 = Ilaenv(1, "Zunmqr", []byte{' '}, m, n, nrhs, -1)
	nb4 = Ilaenv(1, "Zunmrq", []byte{' '}, m, n, nrhs, -1)
	nb = max(nb1, nb2, nb3, nb4)
	lwkopt = max(1, mn+2*n+nb*(n+1), 2*mn+nb*nrhs)
	work.SetRe(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, m, n) {
		err = fmt.Errorf("b.Rows < max(1, m, n): b.Rows=%v, m=%v, n=%v", b.Rows, m, n)
	} else if lwork < (mn+max(2*mn, n+1, mn+nrhs)) && !lquery {
		err = fmt.Errorf("lwork < (mn+max(2*mn, n+1, mn+nrhs)) && !lquery: lwork=%v, mn=%v, nrhs=%v, lquery=%v", lwork, mn, nrhs, lquery)
	}

	if err != nil {
		gltest.Xerbla2("Zgelsy", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(m, n, nrhs) == 0 {
		rank = 0
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum) / Dlamch(Precision)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Scale A, B if max entries outside range [SMLNUM,BIGNUM]
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
		rank = 0
		goto label70
	}

	bnrm = Zlange('M', m, nrhs, b, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Zlascl('G', 0, 0, bnrm, smlnum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = Zlascl('G', 0, 0, bnrm, bignum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 2
	}

	//     Compute QR factorization with column pivoting of A:
	//        A * P = Q * R
	if err = Zgeqp3(m, n, a, jpvt, work.Off(0), work.Off(mn), lwork-mn, rwork); err != nil {
		panic(err)
	}
	wsize = float64(mn) + work.GetRe(mn)

	//     complex workspace: MN+NB*(N+1). real workspace 2*N.
	//     Details of Householder rotations stored in WORK(1:MN).
	//
	//     Determine RANK using incremental condition estimation
	work.Set(ismin-1, cone)
	work.Set(ismax-1, cone)
	smax = a.GetMag(0, 0)
	smin = smax
	if a.GetMag(0, 0) == zero {
		rank = 0
		Zlaset(Full, max(m, n), nrhs, czero, czero, b)
		goto label70
	} else {
		rank = 1
	}

label10:
	;
	if rank < mn {
		i = rank + 1
		sminpr, s1, c1 = Zlaic1(imin, rank, work.Off(ismin-1), smin, a.CVector(0, i-1), a.Get(i-1, i-1))
		smaxpr, s2, c2 = Zlaic1(imax, rank, work.Off(ismax-1), smax, a.CVector(0, i-1), a.Get(i-1, i-1))

		if smaxpr*rcond <= sminpr {
			for i = 1; i <= rank; i++ {
				work.Set(ismin+i-1-1, s1*work.Get(ismin+i-1-1))
				work.Set(ismax+i-1-1, s2*work.Get(ismax+i-1-1))
			}
			work.Set(ismin+rank-1, c1)
			work.Set(ismax+rank-1, c2)
			smin = sminpr
			smax = smaxpr
			rank = rank + 1
			goto label10
		}
	}

	//     complex workspace: 3*MN.
	//
	//     Logically partition R = [ R11 R12 ]
	//                             [  0  R22 ]
	//     where R11 = R(1:RANK,1:RANK)
	//
	//     [R11,R12] = [ T11, 0 ] * Y
	if rank < n {
		if err = Ztzrzf(rank, n, a, work.Off(mn), work.Off(2*mn), lwork-2*mn); err != nil {
			panic(err)
		}
	}

	//     complex workspace: 2*MN.
	//     Details of Householder rotations stored in WORK(MN+1:2*MN)
	//
	//     B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS)
	if err = Zunmqr(Left, ConjTrans, m, nrhs, mn, a, work.Off(0), b, work.Off(2*mn), lwork-2*mn); err != nil {
		panic(err)
	}
	wsize = math.Max(wsize, float64(2*mn)+work.GetRe(2*mn))

	//     complex workspace: 2*MN+NB*NRHS.
	//
	//     B(1:RANK,1:NRHS) := inv(T11) * B(1:RANK,1:NRHS)
	if err = goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, rank, nrhs, cone, a, b); err != nil {
		panic(err)
	}

	for j = 1; j <= nrhs; j++ {
		for i = rank + 1; i <= n; i++ {
			b.Set(i-1, j-1, czero)
		}
	}

	//     B(1:N,1:NRHS) := Y**H * B(1:N,1:NRHS)
	if rank < n {
		if err = Zunmrz(Left, ConjTrans, n, nrhs, rank, n-rank, a, work.Off(mn), b, work.Off(2*mn), lwork-2*mn); err != nil {
			panic(err)
		}
	}

	//     complex workspace: 2*MN+NRHS.
	//
	//     B(1:N,1:NRHS) := P * B(1:N,1:NRHS)
	for j = 1; j <= nrhs; j++ {
		for i = 1; i <= n; i++ {
			work.Set((*jpvt)[i-1]-1, b.Get(i-1, j-1))
		}
		goblas.Zcopy(n, work.Off(0, 1), b.CVector(0, j-1, 1))
	}

	//     complex workspace: N.
	//
	//     Undo scaling
	if iascl == 1 {
		if err = Zlascl('G', 0, 0, anrm, smlnum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Zlascl('U', 0, 0, smlnum, anrm, rank, rank, a); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Zlascl('G', 0, 0, anrm, bignum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Zlascl('U', 0, 0, bignum, anrm, rank, rank, a); err != nil {
			panic(err)
		}
	}
	if ibscl == 1 {
		if err = Zlascl('G', 0, 0, smlnum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	} else if ibscl == 2 {
		if err = Zlascl('G', 0, 0, bignum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	}

label70:
	;
	work.SetRe(0, float64(lwkopt))

	return
}
