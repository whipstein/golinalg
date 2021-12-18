package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgejsv computes the singular value decomposition (SVD) of a real M-by-N
// matrix [A], where M >= N. The SVD of [A] is written as
//
//              [A] = [U] * [SIGMA] * [V]^t,
//
// where [SIGMA] is an N-by-N (M-by-N) matrix which is zero except for its N
// diagonal elements, [U] is an M-by-N (or M-by-M) orthonormal matrix, and
// [V] is an N-by-N orthogonal matrix. The diagonal elements of [SIGMA] are
// the singular values of [A]. The columns of [U] and [V] are the left and
// the right singular vectors of [A], respectively. The matrices [U] and [V]
// are computed and stored in the arrays U and V, respectively. The diagonal
// of [SIGMA] is computed and stored in the array SVA.
// Dgejsv can sometimes compute tiny singular values and their singular vectors much
// more accurately than other SVD routines, see below under Further Details.
func Dgejsv(joba, jobu, jobv, jobr, jobt, jobp byte, m, n int, a *mat.Matrix, sva *mat.Vector, u, v *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int) (info int, err error) {
	var almort, defr, errest, goscal, jracc, kill, l2aber, l2kill, l2pert, l2rank, l2tran, lsvec, noscal, rowpiv, rsvec, transp bool
	var aapp, aaqq, aatmax, aatmin, big, big1, condOk, condr1, condr2, entra, entrat, epsln, maxprj, one, scalem, sconda, sfmin, small, temp1, uscal1, uscal2, xsc, zero float64
	var n1, nr, numrank, p, q, warning int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	lsvec = jobu == 'U' || jobu == 'F'
	jracc = jobv == 'J'
	rsvec = jobv == 'V' || jracc
	rowpiv = joba == 'F' || joba == 'G'
	l2rank = joba == 'R'
	l2aber = joba == 'A'
	errest = joba == 'E' || joba == 'G'
	l2tran = jobt == 'T'
	l2kill = jobr == 'R'
	defr = jobr == 'N'
	l2pert = jobp == 'P'

	if !(rowpiv || l2rank || l2aber || errest || joba == 'C') {
		err = fmt.Errorf("!(rowpiv || l2rank || l2aber || errest || joba == 'C'): joba='%c'", joba)
	} else if !(lsvec || jobu == 'N' || jobu == 'W') {
		err = fmt.Errorf("!(lsvec || jobu == 'N' || jobu == 'W'): jobu='%c'", jobu)
	} else if !(rsvec || jobv == 'N' || jobv == 'W') || (jracc && (!lsvec)) {
		err = fmt.Errorf("!(rsvec || jobv == 'N' || jobv == 'W') || (jracc && (!lsvec)): jobv='%c'", jobv)
	} else if !(l2kill || defr) {
		err = fmt.Errorf("!(l2kill || defr): jobr='%c'", jobr)
	} else if !(l2tran || jobt == 'N') {
		err = fmt.Errorf("!(l2tran || jobt == 'N'): jobt='%c'", jobt)
	} else if !(l2pert || jobp == 'N') {
		err = fmt.Errorf("!(l2pert || jobp == 'N'): jobp='%c'", jobp)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if (n < 0) || (n > m) {
		err = fmt.Errorf("(n < 0) || (n > m): m=%v, n=%v", m, n)
	} else if a.Rows < m {
		err = fmt.Errorf("a.Rows < m: a.Rows=%v, m=%v", a.Rows, m)
	} else if lsvec && (u.Rows < m) {
		err = fmt.Errorf("lsvec && (u.Rows < m): jobu='%c', u.Rows=%v, m=%v", jobu, u.Rows, m)
	} else if rsvec && (v.Rows < n) {
		err = fmt.Errorf("rsvec && (v.Rows < n): jobv='%c', v.Rows=%v, n=%v", jobv, v.Rows, n)
	} else if (!(lsvec || rsvec || errest) && (lwork < max(2, 4*n+1, 2*m+n))) || (!(lsvec || rsvec) && errest && (lwork < max(7, 4*n+n*n, 2*m+n))) || (lsvec && (!rsvec) && (lwork < max(7, 2*m+n, 4*n+1))) || (rsvec && (!lsvec) && (lwork < max(7, 2*m+n, 4*n+1))) || (lsvec && rsvec && (!jracc) && (lwork < max(2*m+n, 6*n+2*n*n))) || (lsvec && rsvec && jracc && lwork < max(2*m+n, 4*n+n*n, 2*n+n*n+6)) {
		err = fmt.Errorf("(!(lsvec || rsvec || errest) && (lwork < max(2, 4*n+1, 2*m+n))) || (!(lsvec || rsvec) && errest && (lwork < max(7, 4*n+n*n, 2*m+n))) || (lsvec && (!rsvec) && (lwork < max(7, 2*m+n, 4*n+1))) || (rsvec && (!lsvec) && (lwork < max(7, 2*m+n, 4*n+1))) || (lsvec && rsvec && (!jracc) && (lwork < max(2*m+n, 6*n+2*n*n))) || (lsvec && rsvec && jracc && lwork < max(2*m+n, 4*n+n*n, 2*n+n*n+6)): jobu='%c', jobv='%c', lwork=%v, m=%v, n=%v", jobu, jobv, lwork, m, n)
	}

	if err != nil {
		//       #:(
		gltest.Xerbla2("Dgejsv", err)
		return
	}

	//     Quick return for void matrix (Y3K safe)
	// #:)
	if (m == 0) || (n == 0) {
		(*iwork)[0] = 0
		work.Set(0, 0)
		return
	}

	//     Determine whether the matrix U should be M x N or M x M
	if lsvec {
		n1 = n
		if jobu == 'F' {
			n1 = m
		}
	}

	//     Set numerical parameters
	//
	//!    NOTE: Make sure DLAMCH() does not fail on the target architecture.
	epsln = Dlamch(Epsilon)
	sfmin = Dlamch(SafeMinimum)
	small = sfmin / epsln
	big = Dlamch(Overflow)
	//     BIG   = ONE / SFMIN
	//
	//     Initialize SVA(1:N) = diag( ||A e_i||_2 )_1^N
	//
	//(!)  If necessary, scale SVA() to protect the largest norm from
	//     overflow. It is possible that this scaling pushes the smallest
	//     column norm left from the underflow threshold (extreme case).

	scalem = one / math.Sqrt(float64(m)*float64(n))
	noscal = true
	goscal = true
	for p = 1; p <= n; p++ {
		aapp = zero
		aaqq = one
		aapp, aaqq = Dlassq(m, a.Off(0, p-1).Vector(), 1, aapp, aaqq)
		if aapp > big {
			err = fmt.Errorf("aapp > big: aapp=%v, big=%v", aapp, big)
			gltest.Xerbla2("Dgejsv", err)
			return
		}
		aaqq = math.Sqrt(aaqq)
		if (aapp < (big / aaqq)) && noscal {
			sva.Set(p-1, aapp*aaqq)
		} else {
			noscal = false
			sva.Set(p-1, aapp*(aaqq*scalem))
			if goscal {
				goscal = false
				sva.Scal(p-1, scalem, 1)
			}
		}
	}

	if noscal {
		scalem = one
	}

	aapp = zero
	aaqq = big
	for p = 1; p <= n; p++ {
		aapp = math.Max(aapp, sva.Get(p-1))
		if sva.Get(p-1) != zero {
			aaqq = math.Min(aaqq, sva.Get(p-1))
		}
	}

	//     Quick return for zero M x N matrix
	// #:)
	if aapp == zero {
		if lsvec {
			Dlaset(Full, m, n1, zero, one, u)
		}
		if rsvec {
			Dlaset(Full, n, n, zero, one, v)
		}
		work.Set(0, one)
		work.Set(1, one)
		if errest {
			work.Set(2, one)
		}
		if lsvec && rsvec {
			work.Set(3, one)
			work.Set(4, one)
		}
		if l2tran {
			work.Set(5, zero)
			work.Set(6, zero)
		}
		(*iwork)[0] = 0
		(*iwork)[1] = 0
		(*iwork)[2] = 0
		return
	}

	//     Issue warning if denormalized column norms detected. Override the
	//     high relative accuracy request. Issue licence to kill columns
	//     (set them to zero) whose norm is less than sigma_max / BIG (roughly).
	// #:(
	warning = 0
	if aaqq <= sfmin {
		l2rank = true
		l2kill = true
		warning = 1
	}

	//     Quick return for one-column matrix
	// #:)
	if n == 1 {

		if lsvec {
			if err = Dlascl('G', 0, 0, sva.Get(0), scalem, m, 1, a.Off(0, 0)); err != nil {
				panic(err)
			}
			Dlacpy(Full, m, 1, a, u)
			//           computing all M left singular vectors of the M x 1 matrix
			if n1 != n {
				if err = Dgeqrf(m, n, u, work, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				if err = Dorgqr(m, n1, 1, u, work, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				u.Off(0, 0).Vector().Copy(m, a.Off(0, 0).Vector(), 1, 1)
			}
		}
		if rsvec {
			v.Set(0, 0, one)
		}
		if sva.Get(0) < (big * scalem) {
			sva.Set(0, sva.Get(0)/scalem)
			scalem = one
		}
		work.Set(0, one/scalem)
		work.Set(1, one)
		if sva.Get(0) != zero {
			(*iwork)[0] = 1
			if (sva.Get(0) / scalem) >= sfmin {
				(*iwork)[1] = 1
			} else {
				(*iwork)[1] = 0
			}
		} else {
			(*iwork)[0] = 0
			(*iwork)[1] = 0
		}
		(*iwork)[2] = 0
		if errest {
			work.Set(2, one)
		}
		if lsvec && rsvec {
			work.Set(3, one)
			work.Set(4, one)
		}
		if l2tran {
			work.Set(5, zero)
			work.Set(6, zero)
		}
		return

	}

	transp = false
	l2tran = l2tran && (m == n)

	aatmax = -one
	aatmin = big
	if rowpiv || l2tran {
		//     Compute the row norms, needed to determine row pivoting sequence
		//     (in the case of heavily row weighted A, row pivoting is strongly
		//     advised) and to collect information needed to compare the
		//     structures of A * A^t and A^t * A (in the case L2TRAN.EQ..TRUE.).
		if l2tran {
			for p = 1; p <= m; p++ {
				xsc = zero
				temp1 = one
				xsc, temp1 = Dlassq(n, a.Off(p-1, 0).Vector(), a.Rows, xsc, temp1)
				//              DLASSQ gets both the ell_2 and the ell_infinity norm
				//              in one pass through the vector
				work.Set(m+n+p-1, xsc*scalem)
				work.Set(n+p-1, xsc*(scalem*math.Sqrt(temp1)))
				aatmax = math.Max(aatmax, work.Get(n+p-1))
				if work.Get(n+p-1) != zero {
					aatmin = math.Min(aatmin, work.Get(n+p-1))
				}
			}
		} else {
			for p = 1; p <= m; p++ {
				work.Set(m+n+p-1, scalem*math.Abs(a.Get(p-1, a.Off(p-1, 1-1).Vector().Iamax(n, a.Rows)-1)))
				aatmax = math.Max(aatmax, work.Get(m+n+p-1))
				aatmin = math.Min(aatmin, work.Get(m+n+p-1))
			}
		}

	}

	//     For square matrix A try to determine whether A^t  would be  better
	//     input for the preconditioned Jacobi SVD, with faster convergence.
	//     The decision is based on an O(N) function of the vector of column
	//     and row norms of A, based on the Shannon entropy. This should give
	//     the right choice in most cases when the difference actually matters.
	//     It may fail and pick the slower converging side.
	entra = zero
	entrat = zero
	if l2tran {

		xsc = zero
		temp1 = one
		xsc, temp1 = Dlassq(n, sva, 1, xsc, temp1)
		temp1 = one / temp1

		entra = zero
		for p = 1; p <= n; p++ {
			big1 = math.Pow(sva.Get(p-1)/xsc, 2) * temp1
			if big1 != zero {
				entra = entra + big1*math.Log(big1)
			}
		}
		entra = -entra / math.Log(float64(n))

		//        Now, SVA().^2/Trace(A^t * A) is a point in the probability simplex.
		//        It is derived from the diagonal of  A^t * A.  Do the same with the
		//        diagonal of A * A^t, compute the entropy of the corresponding
		//        probability distribution. Note that A * A^t and A^t * A have the
		//        same trace.
		entrat = zero
		for p = n + 1; p <= n+m; p++ {
			big1 = math.Pow(work.Get(p-1)/xsc, 2) * temp1
			if big1 != zero {
				entrat = entrat + big1*math.Log(big1)
			}
		}
		entrat = -entrat / math.Log(float64(m))

		//        Analyze the entropies and decide A or A^t. Smaller entropy
		//        usually means better input for the algorithm.
		transp = (entrat < entra)

		//        If A^t is better than A, transpose A.
		if transp {
			//           In an optimal implementation, this trivial transpose
			//           should be replaced with faster transpose.
			for p = 1; p <= n-1; p++ {
				for q = p + 1; q <= n; q++ {
					temp1 = a.Get(q-1, p-1)
					a.Set(q-1, p-1, a.Get(p-1, q-1))
					a.Set(p-1, q-1, temp1)
				}
			}
			for p = 1; p <= n; p++ {
				work.Set(m+n+p-1, sva.Get(p-1))
				sva.Set(p-1, work.Get(n+p-1))
			}
			temp1 = aapp
			aapp = aatmax
			aatmax = temp1
			temp1 = aaqq
			aaqq = aatmin
			aatmin = temp1
			kill = lsvec
			lsvec = rsvec
			rsvec = kill
			if lsvec {
				n1 = n
			}

			rowpiv = true
		}

	}
	//     END IF L2TRAN
	//
	//     Scale the matrix so that its maximal singular value remains less
	//     than DSQRT(BIG) -- the matrix is scaled so that its maximal column
	//     has Euclidean norm equal to DSQRT(BIG/N). The only reason to keep
	//     DSQRT(BIG) instead of BIG is the fact that Dgejsv uses LAPACK and
	//     BLAS routines that, in some implementations, are not capable of
	//     working in the full interval [SFMIN,BIG] and that they may provoke
	//     overflows in the intermediate results. If the singular values spread
	//     from SFMIN to BIG, then DGESVJ will compute them. So, in that case,
	//     one should use DGESVJ instead of Dgejsv.
	big1 = math.Sqrt(big)
	temp1 = math.Sqrt(big / float64(n))

	if err = Dlascl('G', 0, 0, aapp, temp1, n, 1, sva.Matrix(n, opts)); err != nil {
		panic(err)
	}
	if aaqq > (aapp * sfmin) {
		aaqq = (aaqq / aapp) * temp1
	} else {
		aaqq = (aaqq * temp1) / aapp
	}
	temp1 = temp1 * scalem
	if err = Dlascl('G', 0, 0, aapp, temp1, m, n, a); err != nil {
		panic(err)
	}

	//     To undo scaling at the end of this procedure, multiply the
	//     computed singular values with USCAL2 / USCAL1.
	uscal1 = temp1
	uscal2 = aapp

	if l2kill {
		//        L2KILL enforces computation of nonzero singular values in
		//        the restricted range of condition number of the initial A,
		//        sigma_max(A) / sigma_min(A) approx. DSQRT(BIG)/DSQRT(SFMIN).
		xsc = math.Sqrt(sfmin)
	} else {
		xsc = small

		//        Now, if the condition number of A is too big,
		//        sigma_max(A) / sigma_min(A) .GT. DSQRT(BIG/N) * EPSLN / SFMIN,
		//        as a precaution measure, the full SVD is computed using DGESVJ
		//        with accumulated Jacobi rotations. This provides numerically
		//        more robust computation, at the cost of slightly increased run
		//        time. Depending on the concrete implementation of BLAS and LAPACK
		//        (i.e. how they behave in presence of extreme ill-conditioning) the
		//        implementor may decide to remove this switch.
		if (aaqq < math.Sqrt(sfmin)) && lsvec && rsvec {
			jracc = true
		}

	}
	if aaqq < xsc {
		for p = 1; p <= n; p++ {
			if sva.Get(p-1) < xsc {
				Dlaset(Full, m, 1, zero, zero, a.Off(0, p-1))
				sva.Set(p-1, zero)
			}
		}
	}

	//     Preconditioning using QR factorization with pivoting
	if rowpiv {
		//        Optional row permutation (Bjoerck row pivoting):
		//        A result by Cox and Higham shows that the Bjoerck's
		//        row pivoting combined with standard column pivoting
		//        has similar effect as Powell-Reid complete pivoting.
		//        The ell-infinity norms of A are made nonincreasing.
		for p = 1; p <= m-1; p++ {
			q = work.Off(m+n+p-1).Iamax(m-p+1, 1) + p - 1
			(*iwork)[2*n+p-1] = q
			if p != q {
				temp1 = work.Get(m + n + p - 1)
				work.Set(m+n+p-1, work.Get(m+n+q-1))
				work.Set(m+n+q-1, temp1)
			}
		}
		Dlaswp(n, a, 1, m-1, *toSlice(iwork, 2*n), 1)
	}

	//     End of the preparation phase (scaling, optional sorting and
	//     transposing, optional flushing of small columns).
	//
	//     Preconditioning
	//
	//     If the full SVD is needed, the right singular vectors are computed
	//     from a matrix equation, and for that we need theoretical analysis
	//     of the Businger-Golub pivoting. So we use DGEQP3 as the first RR QRF.
	//     In all other cases the first RR QRF can be chosen by other criteria
	//     (eg speed by replacing global with restricted window pivoting, such
	//     as in SGEQPX from TOMS # 782). Good results will be obtained using
	//     SGEQPX with properly (!) chosen numerical parameters.
	//     Any improvement of DGEQP3 improves overal performance of Dgejsv.
	//
	//     A * P1 = Q1 * [ R1^t 0]^t:
	for p = 1; p <= n; p++ {
		//        .. all columns are free columns
		(*iwork)[p-1] = 0
	}
	if err = Dgeqp3(m, n, a, iwork, work, work.Off(n), lwork-n); err != nil {
		panic(err)
	}

	//     The upper triangular matrix R1 from the first QRF is inspected for
	//     rank deficiency and possibilities for deflation, or possible
	//     ill-conditioning. Depending on the user specified flag L2RANK,
	//     the procedure explores possibilities to reduce the numerical
	//     rank by inspecting the computed upper triangular factor. If
	//     L2RANK or L2ABER are up, then Dgejsv will compute the SVD of
	//     A + dA, where ||dA|| <= f(M,N)*EPSLN.
	nr = 1
	if l2aber {
		//        Standard absolute error bound suffices. All sigma_i with
		//        sigma_i < N*EPSLN*||A|| are flushed to zero. This is an
		//        aggressive enforcement of lower numerical rank by introducing a
		//        backward error of the order of N*EPSLN*||A||.
		temp1 = math.Sqrt(float64(n)) * epsln
		for p = 2; p <= n; p++ {
			if math.Abs(a.Get(p-1, p-1)) >= (temp1 * math.Abs(a.Get(0, 0))) {
				nr = nr + 1
			} else {
				goto label3002
			}
		}
	label3002:
	} else if l2rank {
		//        .. similarly as above, only slightly more gentle (less aggressive).
		//        Sudden drop on the diagonal of R1 is used as the criterion for
		//        close-to-rank-deficient.
		temp1 = math.Sqrt(sfmin)
		for p = 2; p <= n; p++ {
			if (math.Abs(a.Get(p-1, p-1)) < (epsln * math.Abs(a.Get(p-1-1, p-1-1)))) || (math.Abs(a.Get(p-1, p-1)) < small) || (l2kill && (math.Abs(a.Get(p-1, p-1)) < temp1)) {
				goto label3402
			}
			nr = nr + 1
		}
	label3402:
	} else {
		//        The goal is high relative accuracy. However, if the matrix
		//        has high scaled condition number the relative accuracy is in
		//        general not feasible. Later on, a condition number estimator
		//        will be deployed to estimate the scaled condition number.
		//        Here we just remove the underflowed part of the triangular
		//        factor. This prevents the situation in which the code is
		//        working hard to get the accuracy not warranted by the data.
		temp1 = math.Sqrt(sfmin)
		for p = 2; p <= n; p++ {
			if (math.Abs(a.Get(p-1, p-1)) < small) || (l2kill && (math.Abs(a.Get(p-1, p-1)) < temp1)) {
				goto label3302
			}
			nr = nr + 1
		}
	label3302:
	}

	almort = false
	if nr == n {
		maxprj = one
		for p = 2; p <= n; p++ {
			temp1 = math.Abs(a.Get(p-1, p-1)) / sva.Get((*iwork)[p-1]-1)
			maxprj = math.Min(maxprj, temp1)
		}
		if math.Pow(maxprj, 2) >= one-float64(n)*epsln {
			almort = true
		}
	}

	sconda = -one
	condr1 = -one
	condr2 = -one

	if errest {
		if n == nr {
			if rsvec {
				//              .. V is available as workspace
				Dlacpy(Upper, n, n, a, v)
				for p = 1; p <= n; p++ {
					temp1 = sva.Get((*iwork)[p-1] - 1)
					v.Off(0, p-1).Vector().Scal(p, one/temp1, 1)
				}
				if temp1, err = Dpocon(Upper, n, v, one, work.Off(n), toSlice(iwork, 2*n+m)); err != nil {
					panic(err)
				}
			} else if lsvec {
				//              .. U is available as workspace
				Dlacpy(Upper, n, n, a, u)
				for p = 1; p <= n; p++ {
					temp1 = sva.Get((*iwork)[p-1] - 1)
					u.Off(0, p-1).Vector().Scal(p, one/temp1, 1)
				}
				if temp1, err = Dpocon(Upper, n, u, one, work.Off(n), toSlice(iwork, 2*n+m)); err != nil {
					panic(err)
				}
			} else {
				Dlacpy(Upper, n, n, a, work.Off(n).Matrix(n, opts))
				for p = 1; p <= n; p++ {
					temp1 = sva.Get((*iwork)[p-1] - 1)
					work.Off(n+(p-1)*n).Scal(p, one/temp1, 1)
				}
				//           .. the columns of R are scaled to have unit Euclidean lengths.
				if temp1, err = Dpocon(Upper, n, work.Off(n).Matrix(n, opts), one, work.Off(n+n*n), toSlice(iwork, 2*n+m)); err != nil {
					panic(err)
				}
			}
			sconda = one / math.Sqrt(temp1)
			//           SCONDA is an estimate of DSQRT(||(R^t * R)^(-1)||_1).
			//           N^(-1/4) * SCONDA <= ||R^(-1)||_2 <= N^(1/4) * SCONDA
		} else {
			sconda = -one
		}
	}

	l2pert = l2pert && (math.Abs(a.Get(0, 0)/a.Get(nr-1, nr-1)) > math.Sqrt(big1))
	//     If there is no violent scaling, artificial perturbation is not needed.
	//
	//     Phase 3:

	if !(rsvec || lsvec) {
		//         Singular Values only
		//
		//         .. transpose A(1:NR,1:N)
		for p = 1; p <= min(n-1, nr); p++ {
			a.Off(p, p-1).Vector().Copy(n-p, a.Off(p-1, p).Vector(), a.Rows, 1)
		}

		//        The following two DO-loops introduce small relative perturbation
		//        into the strict upper triangle of the lower triangular matrix.
		//        Small entries below the main diagonal are also changed.
		//        This modification is useful if the computing environment does not
		//        provide/allow FLUSH TO ZERO underflow, for it prevents many
		//        annoying denormalized numbers in case of strongly scaled matrices.
		//        The perturbation is structured so that it does not introduce any
		//        new perturbation of the singular values, and it does not destroy
		//        the job done by the preconditioner.
		//        The licence for this perturbation is in the variable L2PERT, which
		//        should be .FALSE. if FLUSH TO ZERO underflow is active.
		if !almort {

			if l2pert {
				//              XSC = DSQRT(SMALL)
				xsc = epsln / float64(n)
				for q = 1; q <= nr; q++ {
					temp1 = xsc * math.Abs(a.Get(q-1, q-1))
					for p = 1; p <= n; p++ {
						if ((p > q) && (math.Abs(a.Get(p-1, q-1)) <= temp1)) || (p < q) {
							a.Set(p-1, q-1, math.Copysign(temp1, a.Get(p-1, q-1)))
						}
					}
				}
			} else {
				Dlaset(Upper, nr-1, nr-1, zero, zero, a.Off(0, 1))
			}

			//            .. second preconditioning using the QR factorization
			if err = Dgeqrf(n, nr, a, work, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			//           .. and transpose upper to lower triangular
			for p = 1; p <= nr-1; p++ {
				a.Off(p, p-1).Vector().Copy(nr-p, a.Off(p-1, p).Vector(), a.Rows, 1)
			}

		}

		//           Row-cyclic Jacobi SVD algorithm with column pivoting
		//
		//           .. again some perturbation (a "background noise") is added
		//           to drown denormals
		if l2pert {
			//              XSC = DSQRT(SMALL)
			xsc = epsln / float64(n)
			for q = 1; q <= nr; q++ {
				temp1 = xsc * math.Abs(a.Get(q-1, q-1))
				for p = 1; p <= nr; p++ {
					if ((p > q) && (math.Abs(a.Get(p-1, q-1)) <= temp1)) || (p < q) {
						a.Set(p-1, q-1, math.Copysign(temp1, a.Get(p-1, q-1)))
					}
				}
			}
		} else {
			Dlaset(Upper, nr-1, nr-1, zero, zero, a.Off(0, 1))
		}

		//           .. and one-sided Jacobi rotations are started on a lower
		//           triangular matrix (plus perturbation which is ignored in
		//           the part which destroys triangular form (confusing?!))
		if info, err = Dgesvj('L', 'N', 'N', nr, nr, a, sva, n, v, work, lwork); err != nil {
			panic(err)
		}

		scalem = work.Get(0)
		numrank = int(math.Round(work.Get(1)))

	} else if rsvec && (!lsvec) {
		//        -> Singular Values and Right Singular Vectors <-
		if almort {
			//           .. in this case NR equals N
			for p = 1; p <= nr; p++ {
				v.Off(p-1, p-1).Vector().Copy(n-p+1, a.Off(p-1, p-1).Vector(), a.Rows, 1)
			}
			Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))

			if info, err = Dgesvj('L', 'U', 'N', n, nr, v, sva, nr, a, work, lwork); err != nil {
				panic(err)
			}
			scalem = work.Get(0)
			numrank = int(math.Round(work.Get(1)))
		} else {
			//        .. two more QR factorizations ( one QRF is not enough, two require
			//        accumulated product of Jacobi rotations, three are perfect )
			Dlaset(Lower, nr-1, nr-1, zero, zero, a.Off(1, 0))
			if err = Dgelqf(nr, n, a, work, work.Off(n), lwork-n); err != nil {
				panic(err)
			}
			Dlacpy(Lower, nr, nr, a, v)
			Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
			if err = Dgeqrf(nr, nr, v, work.Off(n), work.Off(2*n), lwork-2*n); err != nil {
				panic(err)
			}
			for p = 1; p <= nr; p++ {
				v.Off(p-1, p-1).Vector().Copy(nr-p+1, v.Off(p-1, p-1).Vector(), v.Rows, 1)
			}
			Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))

			if info, err = Dgesvj('L', 'U', 'N', nr, nr, v, sva, nr, u, work.Off(n), lwork); err != nil {
				panic(err)
			}
			scalem = work.Get(n + 1 - 1)
			numrank = int(math.Round(work.Get(n + 2 - 1)))
			if nr < n {
				Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
				Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
				Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
			}

			if err = Dormlq(Left, Trans, n, n, nr, a, work, v, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

		}

		for p = 1; p <= n; p++ {
			a.Off((*iwork)[p-1]-1, 0).Vector().Copy(n, v.Off(p-1, 0).Vector(), v.Rows, a.Rows)
		}
		Dlacpy(Full, n, n, a, v)

		if transp {
			Dlacpy(Full, n, n, v, u)
		}

	} else if lsvec && (!rsvec) {
		//        .. Singular Values and Left Singular Vectors                 ..
		//
		//        .. second preconditioning step to avoid need to accumulate
		//        Jacobi rotations in the Jacobi iterations.
		for p = 1; p <= nr; p++ {
			u.Off(p-1, p-1).Vector().Copy(n-p+1, a.Off(p-1, p-1).Vector(), a.Rows, 1)
		}
		Dlaset(Upper, nr-1, nr-1, zero, zero, u.Off(0, 1))

		if err = Dgeqrf(n, nr, u, work.Off(n), work.Off(2*n), lwork-2*n); err != nil {
			panic(err)
		}

		for p = 1; p <= nr-1; p++ {
			u.Off(p, p-1).Vector().Copy(nr-p, u.Off(p-1, p).Vector(), u.Rows, 1)
		}
		Dlaset(Upper, nr-1, nr-1, zero, zero, u.Off(0, 1))

		if info, err = Dgesvj('L', 'U', 'N', nr, nr, u, sva, nr, a, work.Off(n), lwork-n); err != nil {
			panic(err)
		}
		scalem = work.Get(n + 1 - 1)
		numrank = int(math.Round(work.Get(n + 2 - 1)))

		if nr < m {
			Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
			if nr < n1 {
				Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
				Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
			}
		}

		if err = Dormqr(Left, NoTrans, m, n1, n, a, work, u, work.Off(n), lwork-n); err != nil {
			panic(err)
		}

		if rowpiv {
			Dlaswp(n1, u, 1, m-1, *toSlice(iwork, 2*n), -1)
		}

		for p = 1; p <= n1; p++ {
			xsc = one / u.Off(0, p-1).Vector().Nrm2(m, 1)
			u.Off(0, p-1).Vector().Scal(m, xsc, 1)
		}

		if transp {
			Dlacpy(Full, n, n, u, v)
		}

	} else {
		//        .. Full SVD ..
		if !jracc {

			if !almort {
				//           Second Preconditioning Step (QRF [with pivoting])
				//           Note that the composition of TRANSPOSE, QRF and TRANSPOSE is
				//           equivalent to an LQF CALL. Since in many libraries the QRF
				//           seems to be better optimized than the LQF, we do explicit
				//           transpose and use the QRF. This is subject to changes in an
				//           optimized implementation of Dgejsv.
				for p = 1; p <= nr; p++ {
					v.Off(p-1, p-1).Vector().Copy(n-p+1, a.Off(p-1, p-1).Vector(), a.Rows, 1)
				}

				//           .. the following two loops perturb small entries to avoid
				//           denormals in the second QR factorization, where they are
				//           as good as zeros. This is done to avoid painfully slow
				//           computation with denormals. The relative size of the perturbation
				//           is a parameter that can be changed by the implementer.
				//           This perturbation device will be obsolete on machines with
				//           properly implemented arithmetic.
				//           To switch it off, set L2PERT=.FALSE. To remove it from  the
				//           code, remove the action under L2PERT=.TRUE., leave the ELSE part.
				//           The following two loops should be blocked and fused with the
				//           transposed copy above.
				if l2pert {
					xsc = math.Sqrt(small)
					for q = 1; q <= nr; q++ {
						temp1 = xsc * math.Abs(v.Get(q-1, q-1))
						for p = 1; p <= n; p++ {
							if (p > q) && (math.Abs(v.Get(p-1, q-1)) <= temp1) || (p < q) {
								v.Set(p-1, q-1, math.Copysign(temp1, v.Get(p-1, q-1)))
							}
							if p < q {
								v.Set(p-1, q-1, -v.Get(p-1, q-1))
							}
						}
					}
				} else {
					Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
				}

				//           Estimate the row scaled condition number of R1
				//           (If R1 is rectangular, N > NR, then the condition number
				//           of the leading NR x NR submatrix is estimated.)
				Dlacpy(Lower, nr, nr, v, work.Off(2*n).Matrix(nr, opts))
				for p = 1; p <= nr; p++ {
					temp1 = work.Off(2*n+(p-1)*nr+p-1).Nrm2(nr-p+1, 1)
					work.Off(2*n+(p-1)*nr+p-1).Scal(nr-p+1, one/temp1, 1)
				}
				if temp1, err = Dpocon(Lower, nr, work.Off(2*n).Matrix(nr, opts), one, work.Off(2*n+nr*nr), toSlice(iwork, m+2*n)); err != nil {
					panic(err)
				}
				condr1 = one / math.Sqrt(temp1)
				//           .. here need a second opinion on the condition number
				//           .. then assume worst case scenario
				//           R1 is OK for inverse <=> CONDR1 .LT. DBLE(N)
				//           more conservative    <=> CONDR1 .LT. DSQRT(DBLE(N))

				condOk = math.Sqrt(float64(nr))
				//[TP]       COND_OK is a tuning parameter.
				if condr1 < condOk {
					//              .. the second QRF without pivoting. Note: in an optimized
					//              implementation, this QRF should be implemented as the QRF
					//              of a lower triangular matrix.
					//              R1^t = Q2 * R2
					if err = Dgeqrf(n, nr, v, work.Off(n), work.Off(2*n), lwork-2*n); err != nil {
						panic(err)
					}
					//
					if l2pert {
						xsc = math.Sqrt(small) / epsln
						for p = 2; p <= nr; p++ {
							for q = 1; q <= p-1; q++ {
								temp1 = xsc * math.Min(math.Abs(v.Get(p-1, p-1)), math.Abs(v.Get(q-1, q-1)))
								if math.Abs(v.Get(q-1, p-1)) <= temp1 {
									v.Set(q-1, p-1, math.Copysign(temp1, v.Get(q-1, p-1)))
								}
							}
						}
					}

					if nr != n {
						Dlacpy(Full, n, nr, v, work.Off(2*n).Matrix(n, opts))
					}
					//              .. save ...
					//
					//           .. this transposed copy should be better than naive
					for p = 1; p <= nr-1; p++ {
						v.Off(p, p-1).Vector().Copy(nr-p, v.Off(p-1, p).Vector(), v.Rows, 1)
					}

					condr2 = condr1

				} else {
					//              .. ill-conditioned case: second QRF with pivoting
					//              Note that windowed pivoting would be equally good
					//              numerically, and more run-time efficient. So, in
					//              an optimal implementation, the next call to DGEQP3
					//              should be replaced with eg. CALL SGEQPX (ACM TOMS #782)
					//              with properly (carefully) chosen parameters.
					//
					//              R1^t * P2 = Q2 * R2
					for p = 1; p <= nr; p++ {
						(*iwork)[n+p-1] = 0
					}
					if err = Dgeqp3(n, nr, v, toSlice(iwork, n), work.Off(n), work.Off(2*n), lwork-2*n); err != nil {
						panic(err)
					}
					//*               CALL DGEQRF( N, NR, V, LDV, WORK(N+1), WORK(2*N+1),
					//*     $              LWORK-2*N, IERR )
					if l2pert {
						xsc = math.Sqrt(small)
						for p = 2; p <= nr; p++ {
							for q = 1; q <= p-1; q++ {
								temp1 = xsc * math.Min(math.Abs(v.Get(p-1, p-1)), math.Abs(v.Get(q-1, q-1)))
								if math.Abs(v.Get(q-1, p-1)) <= temp1 {
									v.Set(q-1, p-1, math.Copysign(temp1, v.Get(q-1, p-1)))
								}
							}
						}
					}

					Dlacpy(Full, n, nr, v, work.Off(2*n).Matrix(n, opts))

					if l2pert {
						xsc = math.Sqrt(small)
						for p = 2; p <= nr; p++ {
							for q = 1; q <= p-1; q++ {
								temp1 = xsc * math.Min(math.Abs(v.Get(p-1, p-1)), math.Abs(v.Get(q-1, q-1)))
								v.Set(p-1, q-1, -math.Copysign(temp1, v.Get(q-1, p-1)))
							}
						}
					} else {
						Dlaset(Lower, nr-1, nr-1, zero, zero, v.Off(1, 0))
					}
					//              Now, compute R2 = L3 * Q3, the LQ factorization.
					if err = Dgelqf(nr, nr, v, work.Off(2*n+n*nr), work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}
					//              .. and estimate the condition number
					Dlacpy(Lower, nr, nr, v, work.Off(2*n+n*nr+nr).Matrix(nr, opts))
					for p = 1; p <= nr; p++ {
						temp1 = work.Off(2*n+n*nr+nr+p-1).Nrm2(p, nr)
						work.Off(2*n+n*nr+nr+p-1).Scal(p, one/temp1, nr)
					}
					if temp1, err = Dpocon(Lower, nr, work.Off(2*n+n*nr+nr).Matrix(nr, opts), one, work.Off(2*n+n*nr+nr+nr*nr), toSlice(iwork, m+2*n)); err != nil {
						panic(err)
					}
					condr2 = one / math.Sqrt(temp1)

					if condr2 >= condOk {
						//                 .. save the Householder vectors used for Q3
						//                 (this overwrites the copy of R2, as it will not be
						//                 needed in this branch, but it does not overwritte the
						//                 Huseholder vectors of Q2.).
						Dlacpy(Upper, nr, nr, v, work.Off(2*n).Matrix(n, opts))
						//                 .. and the rest of the information on Q3 is in
						//                 WORK(2*N+N*NR+1:2*N+N*NR+N)
					}

				}

				if l2pert {
					xsc = math.Sqrt(small)
					for q = 2; q <= nr; q++ {
						temp1 = xsc * v.Get(q-1, q-1)
						for p = 1; p <= q-1; p++ {
							//                    V(p,q) = - DSIGN( TEMP1, V(q,p) )
							v.Set(p-1, q-1, -math.Copysign(temp1, v.Get(p-1, q-1)))
						}
					}
				} else {
					Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
				}

				//        Second preconditioning finished; continue with Jacobi SVD
				//        The input matrix is lower trinagular.
				//
				//        Recover the right singular vectors as solution of a well
				//        conditioned triangular matrix equation.
				if condr1 < condOk {

					if info, err = Dgesvj('L', 'U', 'N', nr, nr, v, sva, nr, u, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}
					scalem = work.Get(2*n + n*nr + nr + 1 - 1)
					numrank = int(math.Round(work.Get(2*n + n*nr + nr + 2 - 1)))
					for p = 1; p <= nr; p++ {
						u.Off(0, p-1).Vector().Copy(nr, v.Off(0, p-1).Vector(), 1, 1)
						v.Off(0, p-1).Vector().Scal(nr, sva.Get(p-1), 1)
					}
					//        .. pick the right matrix equation and solve it

					if nr == n {
						// :))             .. best case, R1 is inverted. The solution of this matrix
						//                 equation is Q2*V2 = the product of the Jacobi rotations
						//                 used in DGESVJ, premultiplied with the orthogonal matrix
						//                 from the second QR factorization.
						err = v.Trsm(Left, Upper, NoTrans, NonUnit, nr, nr, one, a)
					} else {
						//                 .. R1 is well conditioned, but non-square. Transpose(R2)
						//                 is inverted to get the product of the Jacobi rotations
						//                 used in DGESVJ. The Q-factor from the second QR
						//                 factorization is then built in explicitly.
						err = v.Trsm(Left, Upper, Trans, NonUnit, nr, nr, one, work.Off(2*n).Matrix(n, opts))
						if nr < n {
							Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
							Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
							Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
						}
						if err = Dormqr(Left, NoTrans, n, n, nr, work.Off(2*n).Matrix(n, opts), work.Off(n), v, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
							panic(err)
						}
					}

				} else if condr2 < condOk {
					// :)           .. the input matrix A is very likely a relative of
					//              the Kahan matrix :)
					//              The matrix R2 is inverted. The solution of the matrix equation
					//              is Q3^T*V3 = the product of the Jacobi rotations (appplied to
					//              the lower triangular L3 from the LQ factorization of
					//              R2=L3*Q3), pre-multiplied with the transposed Q3.
					if info, err = Dgesvj('L', 'U', 'N', nr, nr, v, sva, nr, u, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}
					scalem = work.Get(2*n + n*nr + nr + 1 - 1)
					numrank = int(math.Round(work.Get(2*n + n*nr + nr + 2 - 1)))
					for p = 1; p <= nr; p++ {
						u.Off(0, p-1).Vector().Copy(nr, v.Off(0, p-1).Vector(), 1, 1)
						u.Off(0, p-1).Vector().Scal(nr, sva.Get(p-1), 1)
					}
					err = u.Trsm(Left, Upper, NoTrans, NonUnit, nr, nr, one, work.Off(2*n).Matrix(n, opts))
					//              .. apply the permutation from the second QR factorization
					for q = 1; q <= nr; q++ {
						for p = 1; p <= nr; p++ {
							work.Set(2*n+n*nr+nr+(*iwork)[n+p-1]-1, u.Get(p-1, q-1))
						}
						for p = 1; p <= nr; p++ {
							u.Set(p-1, q-1, work.Get(2*n+n*nr+nr+p-1))
						}
					}
					if nr < n {
						Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
						Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
						Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
					}
					if err = Dormqr(Left, NoTrans, n, n, nr, work.Off(2*n).Matrix(n, opts), work.Off(n), v, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}
				} else {
					//              Last line of defense.
					// #:(          This is a rather pathological case: no scaled condition
					//              improvement after two pivoted QR factorizations. Other
					//              possibility is that the rank revealing QR factorization
					//              or the condition estimator has failed, or the COND_OK
					//              is set very close to ONE (which is unnecessary). Normally,
					//              this branch should never be executed, but in rare cases of
					//              failure of the RRQR or condition estimator, the last line of
					//              defense ensures that Dgejsv completes the task.
					//              Compute the full SVD of L3 using DGESVJ with explicit
					//              accumulation of Jacobi rotations.
					if info, err = Dgesvj('L', 'U', 'V', nr, nr, v, sva, nr, u, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}
					scalem = work.Get(2*n + n*nr + nr + 1 - 1)
					numrank = int(math.Round(work.Get(2*n + n*nr + nr + 2 - 1)))
					if nr < n {
						Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
						Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
						Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
					}
					if err = Dormqr(Left, NoTrans, n, n, nr, work.Off(2*n).Matrix(n, opts), work.Off(n), v, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}

					if err = Dormlq(Left, Trans, nr, nr, nr, work.Off(2*n).Matrix(n, opts), work.Off(2*n+n*nr), u, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
						panic(err)
					}
					for q = 1; q <= nr; q++ {
						for p = 1; p <= nr; p++ {
							work.Set(2*n+n*nr+nr+(*iwork)[n+p-1]-1, u.Get(p-1, q-1))
						}
						for p = 1; p <= nr; p++ {
							u.Set(p-1, q-1, work.Get(2*n+n*nr+nr+p-1))
						}
					}

				}

				//           Permute the rows of V using the (column) permutation from the
				//           first QRF. Also, scale the columns to make them unit in
				//           Euclidean norm. This applies to all cases.
				temp1 = math.Sqrt(float64(n)) * epsln
				for q = 1; q <= n; q++ {
					for p = 1; p <= n; p++ {
						work.Set(2*n+n*nr+nr+(*iwork)[p-1]-1, v.Get(p-1, q-1))
					}
					for p = 1; p <= n; p++ {
						v.Set(p-1, q-1, work.Get(2*n+n*nr+nr+p-1))
					}
					xsc = one / v.Off(0, q-1).Vector().Nrm2(n, 1)
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						v.Off(0, q-1).Vector().Scal(n, xsc, 1)
					}
				}
				//           At this moment, V contains the right singular vectors of A.
				//           Next, assemble the left singular vector matrix U (M x N).
				if nr < m {
					Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
					if nr < n1 {
						Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
						Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
					}
				}

				//           The Q matrix from the first QRF is built into the left singular
				//           matrix U. This applies to all cases.
				if err = Dormqr(Left, NoTrans, m, n1, n, a, work, u, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				//           The columns of U are normalized. The cost is O(M*N) flops.
				temp1 = math.Sqrt(float64(m)) * epsln
				for p = 1; p <= nr; p++ {
					xsc = one / u.Off(0, p-1).Vector().Nrm2(m, 1)
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						u.Off(0, p-1).Vector().Scal(m, xsc, 1)
					}
				}

				//           If the initial QRF is computed with row pivoting, the left
				//           singular vectors must be adjusted.
				if rowpiv {
					Dlaswp(n1, u, 1, m-1, *toSlice(iwork, 2*n), -1)
				}

			} else {
				//        .. the initial matrix A has almost orthogonal columns and
				//        the second QRF is not needed
				Dlacpy(Upper, n, n, a, work.Off(n).Matrix(n, opts))
				if l2pert {
					xsc = math.Sqrt(small)
					for p = 2; p <= n; p++ {
						temp1 = xsc * work.Get(n+(p-1)*n+p-1)
						for q = 1; q <= p-1; q++ {
							work.Set(n+(q-1)*n+p-1, -math.Copysign(temp1, work.Get(n+(p-1)*n+q-1)))
						}
					}
				} else {
					Dlaset(Lower, n-1, n-1, zero, zero, work.Off(n+2-1).Matrix(n, opts))
				}

				if info, err = Dgesvj('U', 'U', 'N', n, n, work.Off(n).Matrix(n, opts), sva, n, u, work.Off(n+n*n), lwork-n-n*n); err != nil {
					panic(err)
				}

				scalem = work.Get(n + n*n + 1 - 1)
				numrank = int(math.Round(work.Get(n + n*n + 2 - 1)))
				for p = 1; p <= n; p++ {
					u.Off(0, p-1).Vector().Copy(n, work.Off(n+(p-1)*n), 1, 1)
					work.Off(n+(p-1)*n).Scal(n, sva.Get(p-1), 1)
				}

				err = work.Off(n).Matrix(n, opts).Trsm(Left, Upper, NoTrans, NonUnit, n, n, one, a)
				for p = 1; p <= n; p++ {
					v.Off((*iwork)[p-1]-1, 0).Vector().Copy(n, work.Off(n+p-1), n, v.Rows)
				}
				temp1 = math.Sqrt(float64(n)) * epsln
				for p = 1; p <= n; p++ {
					xsc = one / v.Off(0, p-1).Vector().Nrm2(n, 1)
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						v.Off(0, p-1).Vector().Scal(n, xsc, 1)
					}
				}

				//           Assemble the left singular vector matrix U (M x N).
				if n < m {
					Dlaset(Full, m-n, n, zero, zero, u.Off(n, 0))
					if n < n1 {
						Dlaset(Full, n, n1-n, zero, zero, u.Off(0, n))
						Dlaset(Full, m-n, n1-n, zero, one, u.Off(n, n))
					}
				}
				if err = Dormqr(Left, NoTrans, m, n1, n, a, work, u, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				temp1 = math.Sqrt(float64(m)) * epsln
				for p = 1; p <= n1; p++ {
					xsc = one / u.Off(0, p-1).Vector().Nrm2(m, 1)
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						u.Off(0, p-1).Vector().Scal(m, xsc, 1)
					}
				}

				if rowpiv {
					Dlaswp(n1, u, 1, m-1, *toSlice(iwork, 2*n), -1)
				}

			}

			//        end of the  >> almost orthogonal case <<  in the full SVD
		} else {
			//        This branch deploys a preconditioned Jacobi SVD with explicitly
			//        accumulated rotations. It is included as optional, mainly for
			//        experimental purposes. It does perform well, and can also be used.
			//        In this implementation, this branch will be automatically activated
			//        if the  condition number sigma_max(A) / sigma_min(A) is predicted
			//        to be greater than the overflow threshold. This is because the
			//        a posteriori computation of the singular vectors assumes robust
			//        implementation of BLAS and some LAPACK procedures, capable of working
			//        in presence of extreme values. Since that is not always the case, ...
			for p = 1; p <= nr; p++ {
				v.Off(p-1, p-1).Vector().Copy(n-p+1, a.Off(p-1, p-1).Vector(), a.Rows, 1)
			}

			if l2pert {
				xsc = math.Sqrt(small / epsln)
				for q = 1; q <= nr; q++ {
					temp1 = xsc * math.Abs(v.Get(q-1, q-1))
					for p = 1; p <= n; p++ {
						if (p > q) && (math.Abs(v.Get(p-1, q-1)) <= temp1) || (p < q) {
							v.Set(p-1, q-1, math.Copysign(temp1, v.Get(p-1, q-1)))
						}
						if p < q {
							v.Set(p-1, q-1, -v.Get(p-1, q-1))
						}
					}
				}
			} else {
				Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
			}
			if err = Dgeqrf(n, nr, v, work.Off(n), work.Off(2*n), lwork-2*n); err != nil {
				panic(err)
			}
			Dlacpy(Lower, n, nr, v, work.Off(2*n).Matrix(n, opts))

			for p = 1; p <= nr; p++ {
				u.Off(p-1, p-1).Vector().Copy(nr-p+1, v.Off(p-1, p-1).Vector(), v.Rows, 1)
			}
			if l2pert {
				xsc = math.Sqrt(small / epsln)
				for q = 2; q <= nr; q++ {
					for p = 1; p <= q-1; p++ {
						temp1 = xsc * math.Min(math.Abs(u.Get(p-1, p-1)), math.Abs(u.Get(q-1, q-1)))
						u.Set(p-1, q-1, -math.Copysign(temp1, u.Get(q-1, p-1)))
					}
				}
			} else {
				Dlaset(Upper, nr-1, nr-1, zero, zero, u.Off(0, 1))
			}
			if info, err = Dgesvj('G', 'U', 'V', nr, nr, u, sva, n, v, work.Off(2*n+n*nr), lwork-2*n-n*nr); err != nil {
				panic(err)
			}
			scalem = work.Get(2*n + n*nr + 1 - 1)
			numrank = int(math.Round(work.Get(2*n + n*nr + 2 - 1)))
			if nr < n {
				Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
				Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
				Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
			}
			if err = Dormqr(Left, NoTrans, n, n, nr, work.Off(2*n).Matrix(n, opts), work.Off(n), v, work.Off(2*n+n*nr+nr), lwork-2*n-n*nr-nr); err != nil {
				panic(err)
			}

			//           Permute the rows of V using the (column) permutation from the
			//           first QRF. Also, scale the columns to make them unit in
			//           Euclidean norm. This applies to all cases.
			temp1 = math.Sqrt(float64(n)) * epsln
			for q = 1; q <= n; q++ {
				for p = 1; p <= n; p++ {
					work.Set(2*n+n*nr+nr+(*iwork)[p-1]-1, v.Get(p-1, q-1))
				}
				for p = 1; p <= n; p++ {
					v.Set(p-1, q-1, work.Get(2*n+n*nr+nr+p-1))
				}
				xsc = one / v.Off(0, q-1).Vector().Nrm2(n, 1)
				if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
					v.Off(0, q-1).Vector().Scal(n, xsc, 1)
				}
			}

			//           At this moment, V contains the right singular vectors of A.
			//           Next, assemble the left singular vector matrix U (M x N).
			if nr < m {
				Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
				if nr < n1 {
					Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
					Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
				}
			}

			if err = Dormqr(Left, NoTrans, m, n1, n, a, work, u, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if rowpiv {
				Dlaswp(n1, u, 1, m-1, *toSlice(iwork, 2*n), -1)
			}

		}
		if transp {
			//           .. swap U and V because the procedure worked on A^t
			for p = 1; p <= n; p++ {
				v.Off(0, p-1).Vector().Swap(n, u.Off(0, p-1).Vector(), 1, 1)
			}
		}

	}
	//     end of the full SVD
	//
	//     Undo scaling, if necessary (and possible)

	if uscal2 <= (big/sva.Get(0))*uscal1 {
		if err = Dlascl('G', 0, 0, uscal1, uscal2, nr, 1, sva.Matrix(n, opts)); err != nil {
			panic(err)
		}
		uscal1 = one
		uscal2 = one
	}

	if nr < n {
		for p = nr + 1; p <= n; p++ {
			sva.Set(p-1, zero)
		}
	}

	work.Set(0, uscal2*scalem)
	work.Set(1, uscal1)
	if errest {
		work.Set(2, sconda)
	}
	if lsvec && rsvec {
		work.Set(3, condr1)
		work.Set(4, condr2)
	}
	if l2tran {
		work.Set(5, entra)
		work.Set(6, entrat)
	}

	(*iwork)[0] = nr
	(*iwork)[1] = numrank
	(*iwork)[2] = warning

	return
}
