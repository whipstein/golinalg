package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelsd computes the minimum-norm solution to a real linear least
// squares problem:
//     minimize 2-norm(| b - A*x |)
// using the singular value decomposition (SVD) of A. A is an M-by-N
// matrix which may be rank-deficient.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
//
// The problem is solved in three steps:
// (1) Reduce the coefficient matrix A to bidiagonal form with
//     Householder transformations, reducing the original problem
//     into a "bidiagonal least squares problem" (BLS)
// (2) Solve the BLS using a divide and conquer approach.
// (3) Apply back all the Householder transformations to solve
//     the original least squares problem.
//
// The effective rank of A is determined by treating as zero those
// singular values which are less than RCOND times the largest singular
// value.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dgelsd(m, n, nrhs int, a, b *mat.Matrix, s *mat.Vector, rcond float64, work *mat.Vector, lwork int, iwork *[]int) (rank, info int, err error) {
	var lquery bool
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, two, zero float64
	var iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nwork, smlsiz, wlalsd int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input arguments.
	minmn = min(m, n)
	maxmn = max(m, n)
	mnthr = Ilaenv(6, "Dgelsd", []byte{' '}, m, n, nrhs, -1)
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, maxmn) {
		err = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=%v, m=%v, n=%v", b.Rows, m, n)
	}

	smlsiz = Ilaenv(9, "Dgelsd", []byte{' '}, 0, 0, 0, 0)

	//     Compute workspace.
	//     (Note: Comments in the code beginning "Workspace:" describe the
	//     minimal amount of workspace needed at that point in the code,
	//     as well as the preferred amount for good performance.
	//     NB refers to the optimal block size for the immediately
	//     following subroutine, as returned by ILAENV.)
	minwrk = 1
	liwork = 1
	minmn = max(1, minmn)
	nlvl = max(int(math.Log(float64(minmn)/float64(smlsiz+1))/math.Log(two))+1, 0)

	if err == nil {
		maxwrk = 0
		liwork = 3*minmn*nlvl + 11*minmn
		mm = m
		if m >= n && m >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns.
			mm = n
			maxwrk = max(maxwrk, n+n*Ilaenv(1, "Dgeqrf", []byte{' '}, m, n, -1, -1))
			maxwrk = max(maxwrk, n+nrhs*Ilaenv(1, "Dormqr", []byte("LT"), m, nrhs, n, -1))
		}
		if m >= n {
			//           Path 1 - overdetermined or exactly determined.
			maxwrk = max(maxwrk, 3*n+(mm+n)*Ilaenv(1, "Dgebrd", []byte{' '}, mm, n, -1, -1))
			maxwrk = max(maxwrk, 3*n+nrhs*Ilaenv(1, "Dormbr", []byte("QLT"), mm, nrhs, n, -1))
			maxwrk = max(maxwrk, 3*n+(n-1)*Ilaenv(1, "Dormbr", []byte("PLN"), n, nrhs, n, -1))
			wlalsd = 9*n + 2*n*smlsiz + 8*n*nlvl + n*nrhs + int(math.Pow(float64(smlsiz+1), 2))
			maxwrk = max(maxwrk, 3*n+wlalsd)
			minwrk = max(3*n+mm, 3*n+nrhs, 3*n+wlalsd)
		}
		if n > m {
			wlalsd = (9 * m) + 2*m*smlsiz + 8*m*nlvl + m*nrhs + int(math.Pow(float64(smlsiz+1), 2))
			if n >= mnthr {
				//              Path 2a - underdetermined, with many more columns
				//              than rows.
				maxwrk = m + m*Ilaenv(1, "Dgelqf", []byte{' '}, m, n, -1, -1)
				maxwrk = max(maxwrk, m*m+4*m+2*m*Ilaenv(1, "Dgebrd", []byte{' '}, m, m, -1, -1))
				maxwrk = max(maxwrk, m*m+4*m+nrhs*Ilaenv(1, "Dormbr", []byte("QLT"), m, nrhs, m, -1))
				maxwrk = max(maxwrk, m*m+4*m+(m-1)*Ilaenv(1, "Dormbr", []byte("PLN"), m, nrhs, m, -1))
				if nrhs > 1 {
					maxwrk = max(maxwrk, m*m+m+m*nrhs)
				} else {
					maxwrk = max(maxwrk, m*m+2*m)
				}
				maxwrk = max(maxwrk, m+nrhs*Ilaenv(1, "Dormlq", []byte("LT"), n, nrhs, m, -1))
				maxwrk = max(maxwrk, m*m+4*m+wlalsd)
				//!     XXX: Ensure the Path 2a case below is triggered.  The workspace

				//!     calculation should use queries for all routines eventually.

				maxwrk = max(maxwrk, 4*m+m*m+max(m, 2*m-4, nrhs, n-3*m))
			} else {
				//              Path 2 - remaining underdetermined cases.
				maxwrk = 3*m + (n+m)*Ilaenv(1, "Dgebrd", []byte{' '}, m, n, -1, -1)
				maxwrk = max(maxwrk, 3*m+nrhs*Ilaenv(1, "Dormbr", []byte("QLT"), m, nrhs, n, -1))
				maxwrk = max(maxwrk, 3*m+m*Ilaenv(1, "Dormbr", []byte("PLN"), n, nrhs, m, -1))
				maxwrk = max(maxwrk, 3*m+wlalsd)
			}
			minwrk = max(3*m+nrhs, 3*m+m, 3*m+wlalsd)
		}
		minwrk = min(minwrk, maxwrk)
		work.Set(0, float64(maxwrk))
		(*iwork)[0] = liwork
		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgelsd", err)
		return
	} else if lquery {
		goto label10
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		rank = 0
		return
	}

	//     Get machine parameters.
	eps = Dlamch(Precision)
	sfmin = Dlamch(SafeMinimum)
	smlnum = sfmin / eps
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	//     Scale A if max entry outside range [SMLNUM,BIGNUM].
	anrm = Dlange('M', m, n, a, work)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM.
		if err = Dlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		if err = Dlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 2
	} else if anrm == zero {
		//        Off all zero. Return zero solution.
		Dlaset(Full, max(m, n), nrhs, zero, zero, b)
		Dlaset(Full, minmn, 1, zero, zero, s.Matrix(1, opts))
		rank = 0
		goto label10
	}

	//     Scale B if max entry outside range [SMLNUM,BIGNUM].
	bnrm = Dlange('M', m, nrhs, b, work)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM.
		if err = Dlascl('G', 0, 0, bnrm, smlnum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		if err = Dlascl('G', 0, 0, bnrm, bignum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 2
	}

	//     If M < N make sure certain entries of B are zero.
	if m < n {
		Dlaset(Full, n-m, nrhs, zero, zero, b.Off(m, 0))
	}

	//     Overdetermined case.
	if m >= n {
		//        Path 1 - overdetermined or exactly determined.
		mm = m
		if m >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns.
			mm = n
			itau = 1
			nwork = itau + n

			//           Compute A=Q*R.
			//           (Workspace: need 2*N, prefer N+N*NB)
			if err = Dgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}

			//           Multiply B by transpose(Q).
			//           (Workspace: need N+NRHS, prefer N+NRHS*NB)
			if err = Dormqr(Left, Trans, m, nrhs, n, a, work.Off(itau-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}

			//           Zero out below R.
			if n > 1 {
				Dlaset(Lower, n-1, n-1, zero, zero, a.Off(1, 0))
			}
		}

		ie = 1
		itauq = ie + n
		itaup = itauq + n
		nwork = itaup + n

		//        Bidiagonalize R in A.
		//        (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB)
		if err = Dgebrd(mm, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of R.
		//        (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB)
		if err = Dormbr('Q', Left, Trans, mm, nrhs, n, a, work.Off(itauq-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Solve the bidiagonal least squares problem.
		if rank, info, err = Dlalsd(Upper, smlsiz, n, nrhs, s, work.Off(ie-1), b, rcond, work.Off(nwork-1), iwork); err != nil {
			panic(err)
		}
		if info != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of R.
		if err = Dormbr('P', Left, NoTrans, n, nrhs, n, a, work.Off(itaup-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

	} else if n >= mnthr && lwork >= 4*m+m*m+max(m, 2*m-4, nrhs, n-3*m, wlalsd) {
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm.
		ldwork = m
		if lwork >= max(4*m+m*a.Rows+max(m, 2*m-4, nrhs, n-3*m), m*a.Rows+m+m*nrhs, 4*m+m*a.Rows+wlalsd) {
			ldwork = a.Rows
		}
		itau = 1
		nwork = m + 1

		//        Compute A=L*Q.
		//        (Workspace: need 2*M, prefer M+M*NB)
		if err = Dgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}
		il = nwork

		//        Copy L to WORK(IL), zeroing out above its diagonal.
		Dlacpy(Lower, m, m, a, work.MatrixOff(il-1, ldwork, opts))
		Dlaset(Upper, m-1, m-1, zero, zero, work.MatrixOff(il+ldwork-1, ldwork, opts))
		ie = il + ldwork*m
		itauq = ie + m
		itaup = itauq + m
		nwork = itaup + m

		//        Bidiagonalize L in WORK(IL).
		//        (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB)
		if err = Dgebrd(m, m, work.MatrixOff(il-1, ldwork, opts), s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of L.
		//        (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
		if err = Dormbr('Q', Left, Trans, m, nrhs, m, work.MatrixOff(il-1, ldwork, opts), work.Off(itauq-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Solve the bidiagonal least squares problem.
		if rank, info, err = Dlalsd(Upper, smlsiz, m, nrhs, s, work.Off(ie-1), b, rcond, work.Off(nwork-1), iwork); err != nil {
			panic(err)
		}
		if info != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of L.
		if err = Dormbr('P', Left, NoTrans, m, nrhs, m, work.MatrixOff(il-1, ldwork, opts), work.Off(itaup-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Zero out below first M rows of B.
		Dlaset(Full, n-m, nrhs, zero, zero, b.Off(m, 0))
		nwork = itau + m

		//        Multiply transpose(Q) by B.
		//        (Workspace: need M+NRHS, prefer M+NRHS*NB)
		if err = Dormlq(Left, Trans, n, nrhs, m, a, work.Off(itau-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

	} else {
		//        Path 2 - remaining underdetermined cases.
		ie = 1
		itauq = ie + m
		itaup = itauq + m
		nwork = itaup + m

		//        Bidiagonalize A.
		//        (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
		if err = Dgebrd(m, n, a, s, work.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors.
		//        (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB)
		if err = Dormbr('Q', Left, Trans, m, nrhs, n, a, work.Off(itauq-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Solve the bidiagonal least squares problem.
		if rank, info, err = Dlalsd(Lower, smlsiz, m, nrhs, s, work.Off(ie-1), b, rcond, work.Off(nwork-1), iwork); err != nil {
			panic(err)
		}
		if info != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of A.
		if err = Dormbr('P', Left, NoTrans, n, nrhs, m, a, work.Off(itaup-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

	}

	//     Undo scaling.
	if iascl == 1 {
		if err = Dlascl('G', 0, 0, anrm, smlnum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Dlascl('G', 0, 0, anrm, bignum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, bignum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	}
	if ibscl == 1 {
		if err = Dlascl('G', 0, 0, smlnum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	} else if ibscl == 2 {
		if err = Dlascl('G', 0, 0, bignum, bnrm, n, nrhs, b); err != nil {
			panic(err)
		}
	}

label10:
	;
	work.Set(0, float64(maxwrk))
	(*iwork)[0] = liwork

	return
}
