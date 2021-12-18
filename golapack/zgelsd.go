package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelsd computes the minimum-norm solution to a real linear least
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
func Zgelsd(m, n, nrhs int, a, b *mat.CMatrix, s *mat.Vector, rcond float64, work *mat.CVector, lwork int, rwork *mat.Vector, iwork *[]int) (rank, info int, err error) {
	var lquery bool
	var czero complex128
	var anrm, bignum, bnrm, eps, one, sfmin, smlnum, two, zero float64
	var iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, lrwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nrwork, nwork, smlsiz int

	zero = 0.0
	one = 1.0
	two = 2.0
	czero = (0.0 + 0.0*1i)

	//     Test the input arguments.
	minmn = min(m, n)
	maxmn = max(m, n)
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
		err = fmt.Errorf("b.Rows < max(1, maxmn): b.Rows=%v, maxmn=%v", b.Rows, maxmn)
	}

	//     Compute workspace.
	//     (Note: Comments in the code beginning "Workspace:" describe the
	//     minimal amount of workspace needed at that point in the code,
	//     as well as the preferred amount for good performance.
	//     NB refers to the optimal block size for the immediately
	//     following subroutine, as returned by ILAENV.)
	if err == nil {
		minwrk = 1
		maxwrk = 1
		liwork = 1
		lrwork = 1
		if minmn > 0 {
			smlsiz = Ilaenv(9, "Zgelsd", []byte{' '}, 0, 0, 0, 0)
			mnthr = Ilaenv(6, "Zgelsd", []byte{' '}, m, n, nrhs, -1)
			nlvl = max(int(math.Log(float64(minmn)/float64(smlsiz+1))/math.Log(two))+1, 0)
			liwork = 3*minmn*nlvl + 11*minmn
			mm = m
			if m >= n && m >= mnthr {
				//              Path 1a - overdetermined, with many more rows than
				//                        columns.
				mm = n
				maxwrk = max(maxwrk, n*Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, -1, -1))
				maxwrk = max(maxwrk, nrhs*Ilaenv(1, "Zunmqr", []byte("LC"), m, nrhs, n, -1))
			}
			if m >= n {
				//              Path 1 - overdetermined or exactly determined.
				lrwork = 10*n + 2*n*smlsiz + 8*n*nlvl + 3*smlsiz*nrhs + max(int(math.Pow(float64(smlsiz+1), 2)), n*(1+nrhs)+2*nrhs)
				maxwrk = max(maxwrk, 2*n+(mm+n)*Ilaenv(1, "Zgebrd", []byte{' '}, mm, n, -1, -1))
				maxwrk = max(maxwrk, 2*n+nrhs*Ilaenv(1, "Zunmbr", []byte("QLC"), mm, nrhs, n, -1))
				maxwrk = max(maxwrk, 2*n+(n-1)*Ilaenv(1, "Zunmbr", []byte("PLN"), n, nrhs, n, -1))
				maxwrk = max(maxwrk, 2*n+n*nrhs)
				minwrk = max(2*n+mm, 2*n+n*nrhs)
			}
			if n > m {
				lrwork = 10*m + 2*m*smlsiz + 8*m*nlvl + 3*smlsiz*nrhs + max(int(math.Pow(float64(smlsiz+1), 2)), n*(1+nrhs)+2*nrhs)
				if n >= mnthr {
					//                 Path 2a - underdetermined, with many more columns
					//                           than rows.
					maxwrk = m + m*Ilaenv(1, "Zgelqf", []byte{' '}, m, n, -1, -1)
					maxwrk = max(maxwrk, m*m+4*m+2*m*Ilaenv(1, "Zgebrd", []byte{' '}, m, m, -1, -1))
					maxwrk = max(maxwrk, m*m+4*m+nrhs*Ilaenv(1, "Zunmbr", []byte("QLC"), m, nrhs, m, -1))
					maxwrk = max(maxwrk, m*m+4*m+(m-1)*Ilaenv(1, "Zunmlq", []byte("LC"), n, nrhs, m, -1))
					if nrhs > 1 {
						maxwrk = max(maxwrk, m*m+m+m*nrhs)
					} else {
						maxwrk = max(maxwrk, m*m+2*m)
					}
					maxwrk = max(maxwrk, m*m+4*m+m*nrhs)
					//!     XXX: Ensure the Path 2a case below is triggered.  The workspace

					//!     calculation should use queries for all routines eventually.

					maxwrk = max(maxwrk, 4*m+m*m+max(m, 2*m-4, nrhs, n-3*m))
				} else {
					//                 Path 2 - underdetermined.
					maxwrk = 2*m + (n+m)*Ilaenv(1, "Zgebrd", []byte{' '}, m, n, -1, -1)
					maxwrk = max(maxwrk, 2*m+nrhs*Ilaenv(1, "Zunmbr", []byte("QLC"), m, nrhs, m, -1))
					maxwrk = max(maxwrk, 2*m+m*Ilaenv(1, "Zunmbr", []byte("PLN"), n, nrhs, m, -1))
					maxwrk = max(maxwrk, 2*m+m*nrhs)
				}
				minwrk = max(2*m+n, 2*m+m*nrhs)
			}
		}
		minwrk = min(minwrk, maxwrk)
		work.SetRe(0, float64(maxwrk))
		(*iwork)[0] = liwork
		rwork.Set(0, float64(lrwork))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgelsd", err)
		return
	} else if lquery {
		return
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
	anrm = Zlange('M', m, n, a, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = Zlascl('G', 0, 0, anrm, smlnum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		if err = Zlascl('G', 0, 0, anrm, bignum, m, n, a); err != nil {
			panic(err)
		}
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Zlaset(Full, max(m, n), nrhs, czero, czero, b)
		Dlaset(Full, minmn, 1, zero, zero, s.Matrix(1, opts))
		rank = 0
		goto label10
	}

	//     Scale B if max entry outside range [SMLNUM,BIGNUM].
	bnrm = Zlange('M', m, nrhs, b, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM.
		if err = Zlascl('G', 0, 0, bnrm, smlnum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM.
		if err = Zlascl('G', 0, 0, bnrm, bignum, m, nrhs, b); err != nil {
			panic(err)
		}
		ibscl = 2
	}

	//     If M < N make sure B(M+1:N,:) = 0
	if m < n {
		Zlaset(Full, n-m, nrhs, czero, czero, b.Off(m, 0))
	}

	//     Overdetermined case.
	if m >= n {
		//        Path 1 - overdetermined or exactly determined.
		mm = m
		if m >= mnthr {
			//           Path 1a - overdetermined, with many more rows than columns
			mm = n
			itau = 1
			nwork = itau + n

			//           Compute A=Q*R.
			//           (RWorkspace: need N)
			//           (CWorkspace: need N, prefer N*NB)
			if err = Zgeqrf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}

			//           Multiply B by transpose(Q).
			//           (RWorkspace: need N)
			//           (CWorkspace: need NRHS, prefer NRHS*NB)
			if err = Zunmqr(Left, ConjTrans, m, nrhs, n, a, work.Off(itau-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
				panic(err)
			}

			//           Zero out below R.
			if n > 1 {
				Zlaset(Lower, n-1, n-1, czero, czero, a.Off(1, 0))
			}
		}

		itauq = 1
		itaup = itauq + n
		nwork = itaup + n
		ie = 1
		nrwork = ie + n

		//        Bidiagonalize R in A.
		//        (RWorkspace: need N)
		//        (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
		if err = Zgebrd(mm, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of R.
		//        (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
		if err = Zunmbr('Q', Left, ConjTrans, mm, nrhs, n, a, work.Off(itauq-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Solve the bidiagonal least squares problem.
		if rank, info, err = Zlalsd(Upper, smlsiz, n, nrhs, s, rwork.Off(ie-1), b, rcond, work.Off(nwork-1), rwork.Off(nrwork-1), iwork); err != nil || info != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of R.
		if err = Zunmbr('P', Left, NoTrans, n, nrhs, n, a, work.Off(itaup-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

	} else if n >= mnthr && lwork >= 4*m+m*m+max(m, 2*m-4, nrhs, n-3*m) {
		//        Path 2a - underdetermined, with many more columns than rows
		//        and sufficient workspace for an efficient algorithm.
		ldwork = m
		if lwork >= max(4*m+m*a.Rows+max(m, 2*m-4, nrhs, n-3*m), m*a.Rows+m+m*nrhs) {
			ldwork = a.Rows
		}
		itau = 1
		nwork = m + 1

		//        Compute A=L*Q.
		//        (CWorkspace: need 2*M, prefer M+M*NB)
		if err = Zgelqf(m, n, a, work.Off(itau-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}
		il = nwork

		//        Copy L to WORK(IL), zeroing out above its diagonal.
		Zlacpy(Lower, m, m, a, work.Off(il-1).CMatrix(ldwork, opts))
		Zlaset(Upper, m-1, m-1, czero, czero, work.Off(il+ldwork-1).CMatrix(ldwork, opts))
		itauq = il + ldwork*m
		itaup = itauq + m
		nwork = itaup + m
		ie = 1
		nrwork = ie + m

		//        Bidiagonalize L in WORK(IL).
		//        (RWorkspace: need M)
		//        (CWorkspace: need M*M+4*M, prefer M*M+4*M+2*M*NB)
		if err = Zgebrd(m, m, work.Off(il-1).CMatrix(ldwork, opts), s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors of L.
		//        (CWorkspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
		if err = Zunmbr('Q', Left, ConjTrans, m, nrhs, m, work.Off(il-1).CMatrix(ldwork, opts), work.Off(itauq-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Solve the bidiagonal least squares problem.
		if rank, info, err = Zlalsd(Upper, smlsiz, m, nrhs, s, rwork.Off(ie-1), b, rcond, work.Off(nwork-1), rwork.Off(nrwork-1), iwork); err != nil || info != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of L.
		if err = Zunmbr('P', Left, NoTrans, m, nrhs, m, work.Off(il-1).CMatrix(ldwork, opts), work.Off(itaup-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Zero out below first M rows of B.
		Zlaset(Full, n-m, nrhs, czero, czero, b.Off(m, 0))
		nwork = itau + m

		//        Multiply transpose(Q) by B.
		//        (CWorkspace: need NRHS, prefer NRHS*NB)
		if err = Zunmlq(Left, ConjTrans, n, nrhs, m, a, work.Off(itau-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

	} else {
		//        Path 2 - remaining underdetermined cases.
		itauq = 1
		itaup = itauq + m
		nwork = itaup + m
		ie = 1
		nrwork = ie + m

		//        Bidiagonalize A.
		//        (RWorkspace: need M)
		//        (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
		if err = Zgebrd(m, n, a, s, rwork.Off(ie-1), work.Off(itauq-1), work.Off(itaup-1), work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Multiply B by transpose of left bidiagonalizing vectors.
		//        (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
		if err = Zunmbr('Q', Left, ConjTrans, m, nrhs, n, a, work.Off(itauq-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

		//        Solve the bidiagonal least squares problem.
		if rank, info, err = Zlalsd(Lower, smlsiz, m, nrhs, s, rwork.Off(ie-1), b, rcond, work.Off(nwork-1), rwork.Off(nrwork-1), iwork); err != nil || info != 0 {
			goto label10
		}

		//        Multiply B by right bidiagonalizing vectors of A.
		if err = Zunmbr('P', Left, NoTrans, n, nrhs, m, a, work.Off(itaup-1), b, work.Off(nwork-1), lwork-nwork+1); err != nil {
			panic(err)
		}

	}

	//     Undo scaling.
	if iascl == 1 {
		if err = Zlascl('G', 0, 0, anrm, smlnum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, smlnum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
			panic(err)
		}
	} else if iascl == 2 {
		if err = Zlascl('G', 0, 0, anrm, bignum, n, nrhs, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, bignum, anrm, minmn, 1, s.Matrix(minmn, opts)); err != nil {
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

label10:
	;
	work.SetRe(0, float64(maxwrk))
	(*iwork)[0] = liwork
	rwork.Set(0, float64(lrwork))

	return
}
