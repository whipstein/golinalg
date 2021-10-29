package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgsen reorders the generalized Schur decomposition of a complex
// matrix pair (A, B) (in terms of an unitary equivalence trans-
// formation Q**H * (A, B) * Z), so that a selected cluster of eigenvalues
// appears in the leading diagonal blocks of the pair (A,B). The leading
// columns of Q and Z form unitary bases of the corresponding left and
// right eigenspaces (deflating subspaces). (A, B) must be in
// generalized Schur canonical form, that is, A and B are both upper
// triangular.
//
// Ztgsen also computes the generalized eigenvalues
//
//          w(j)= ALPHA(j) / BETA(j)
//
// of the reordered matrix pair (A, B).
//
// Optionally, the routine computes estimates of reciprocal condition
// numbers for eigenvalues and eigenspaces. These are Difu[(A11,B11),
// (A22,B22)] and Difl[(A11,B11), (A22,B22)], i.e. the separation(s)
// between the matrix pairs (A11, B11) and (A22,B22) that correspond to
// the selected cluster and the eigenvalues outside the cluster, resp.,
// and norms of "projections" onto left and right eigenspaces w.r.t.
// the selected cluster in the (1,1)-block.
func Ztgsen(ijob int, wantq, wantz bool, _select []bool, n int, a, b *mat.CMatrix, alpha, beta *mat.CVector, q, z *mat.CMatrix, dif *mat.Vector, work *mat.CVector, lwork int, iwork *[]int, liwork int) (m int, pl, pr float64, info int, err error) {
	var lquery, swap, wantd, wantd1, wantd2, wantp bool
	var temp1, temp2 complex128
	var dscale, dsum, one, rdscal, safmin, zero float64
	var i, idifjb, ierr, ijb, k, kase, ks, liwmin, lwmin, mn2, n1, n2 int

	isave := make([]int, 3)

	idifjb = 3
	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters
	lquery = (lwork == -1 || liwork == -1)

	if ijob < 0 || ijob > 5 {
		err = fmt.Errorf("ijob < 0 || ijob > 5: ijob=%v", ijob)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if q.Rows < 1 || (wantq && q.Rows < n) {
		err = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < n): wantq=%v, q.Rows=%v, n=%v", wantq, q.Rows, n)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): wantz=%v, z.Rows=%v, n=%v", wantz, z.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztgsen", err)
		return
	}

	ierr = 0

	wantp = ijob == 1 || ijob >= 4
	wantd1 = ijob == 2 || ijob == 4
	wantd2 = ijob == 3 || ijob == 5
	wantd = wantd1 || wantd2

	//     Set M to the dimension of the specified pair of deflating
	//     subspaces.
	m = 0
	if !lquery || ijob != 0 {
		for k = 1; k <= n; k++ {
			alpha.Set(k-1, a.Get(k-1, k-1))
			beta.Set(k-1, b.Get(k-1, k-1))
			if k < n {
				if _select[k-1] {
					m = m + 1
				}
			} else {
				if _select[n-1] {
					m = m + 1
				}
			}
		}
	}

	if ijob == 1 || ijob == 2 || ijob == 4 {
		lwmin = max(1, 2*m*(n-m))
		liwmin = max(1, n+2)
	} else if ijob == 3 || ijob == 5 {
		lwmin = max(1, 4*m*(n-m))
		liwmin = max(1, 2*m*(n-m), n+2)
	} else {
		lwmin = 1
		liwmin = 1
	}

	work.SetRe(0, float64(lwmin))
	(*iwork)[0] = liwmin

	if lwork < lwmin && !lquery {
		err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
	} else if liwork < liwmin && !lquery {
		err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
	}

	if err != nil {
		gltest.Xerbla2("Ztgsen", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible.
	if m == n || m == 0 {
		if wantp {
			pl = one
			pr = one
		}
		if wantd {
			dscale = zero
			dsum = one
			for i = 1; i <= n; i++ {
				dscale, dsum = Zlassq(n, a.CVector(0, i-1, 1), dscale, dsum)
				dscale, dsum = Zlassq(n, b.CVector(0, i-1, 1), dscale, dsum)
			}
			dif.Set(0, dscale*math.Sqrt(dsum))
			dif.Set(1, dif.Get(0))
		}
		goto label70
	}

	//     Get machine constant
	safmin = Dlamch(SafeMinimum)

	//     Collect the selected blocks at the top-left corner of (A, B).
	ks = 0
	for k = 1; k <= n; k++ {
		swap = _select[k-1]
		if swap {
			ks = ks + 1

			//           Swap the K-th block to position KS. Compute unitary Q
			//           and Z that will swap adjacent diagonal blocks in (A, B).
			if k != ks {
				if ks, ierr, err = Ztgexc(wantq, wantz, n, a, b, q, z, k, ks); err != nil {
					panic(err)
				}
			}

			if ierr > 0 {
				//              Swap is rejected: exit.
				info = 1
				if wantp {
					pl = zero
					pr = zero
				}
				if wantd {
					dif.Set(0, zero)
					dif.Set(1, zero)
				}
				goto label70
			}
		}
	}
	if wantp {
		//        Solve generalized Sylvester equation for R and L:
		//                   A11 * R - L * A22 = A12
		//                   B11 * R - L * B22 = B12
		n1 = m
		n2 = n - m
		i = n1 + 1
		Zlacpy(Full, n1, n2, a.Off(0, i-1), work.CMatrix(n1, opts))
		Zlacpy(Full, n1, n2, b.Off(0, i-1), work.CMatrixOff(n1*n2, n1, opts))
		ijb = 0
		if dscale, *dif.GetPtr(0), ierr, err = Ztgsyl(NoTrans, ijb, n1, n2, a, a.Off(i-1, i-1), work.CMatrix(n1, opts), b, b.Off(i-1, i-1), work.CMatrixOff(n1*n2, n1, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
			panic(err)
		}

		//        Estimate the reciprocal of norms of "projections" onto
		//        left and right eigenspaces
		rdscal = zero
		dsum = one
		rdscal, dsum = Zlassq(n1*n2, work.Off(0, 1), rdscal, dsum)
		pl = rdscal * math.Sqrt(dsum)
		if pl == zero {
			pl = one
		} else {
			pl = dscale / (math.Sqrt(dscale*dscale/pl+pl) * math.Sqrt(pl))
		}
		rdscal = zero
		dsum = one
		rdscal, dsum = Zlassq(n1*n2, work.Off(n1*n2, 1), rdscal, dsum)
		pr = rdscal * math.Sqrt(dsum)
		if pr == zero {
			pr = one
		} else {
			pr = dscale / (math.Sqrt(dscale*dscale/pr+pr) * math.Sqrt(pr))
		}
	}
	if wantd {
		//        Compute estimates Difu and Difl.
		if wantd1 {
			n1 = m
			n2 = n - m
			i = n1 + 1
			ijb = idifjb

			//           Frobenius norm-based Difu estimate.
			if dscale, *dif.GetPtr(0), ierr, err = Ztgsyl(NoTrans, ijb, n1, n2, a, a.Off(i-1, i-1), work.CMatrix(n1, opts), b, b.Off(i-1, i-1), work.CMatrixOff(n1*n2, n1, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
				panic(err)
			}

			//           Frobenius norm-based Difl estimate.
			if dscale, *dif.GetPtr(1), ierr, err = Ztgsyl(NoTrans, ijb, n2, n1, a.Off(i-1, i-1), a, work.CMatrix(n2, opts), b.Off(i-1, i-1), b, work.CMatrixOff(n1*n2, n2, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
				panic(err)
			}
		} else {
			//           Compute 1-norm-based estimates of Difu and Difl using
			//           reversed communication with ZLACN2. In each step a
			//           generalized Sylvester equation or a transposed variant
			//           is solved.
			kase = 0
			n1 = m
			n2 = n - m
			i = n1 + 1
			ijb = 0
			mn2 = 2 * n1 * n2

			//           1-norm-based estimate of Difu.
		label40:
			;
			*dif.GetPtr(0), kase = Zlacn2(mn2, work.Off(mn2), work, dif.Get(0), kase, &isave)
			if kase != 0 {
				if kase == 1 {
					//                 Solve generalized Sylvester equation
					if dscale, *dif.GetPtr(0), ierr, err = Ztgsyl(NoTrans, ijb, n1, n2, a, a.Off(i-1, i-1), work.CMatrix(n1, opts), b, b.Off(i-1, i-1), work.CMatrixOff(n1*n2, n1, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
						panic(err)
					}
				} else {
					//                 Solve the transposed variant.
					if dscale, *dif.GetPtr(0), ierr, err = Ztgsyl(ConjTrans, ijb, n1, n2, a, a.Off(i-1, i-1), work.CMatrix(n1, opts), b, b.Off(i-1, i-1), work.CMatrixOff(n1*n2, n1, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
						panic(err)
					}
				}
				goto label40
			}
			dif.Set(0, dscale/dif.Get(0))
			//
			//           1-norm-based estimate of Difl.
			//
		label50:
			;
			*dif.GetPtr(1), kase = Zlacn2(mn2, work.Off(mn2), work, dif.Get(1), kase, &isave)
			if kase != 0 {
				if kase == 1 {
					//                 Solve generalized Sylvester equation
					if dscale, *dif.GetPtr(1), ierr, err = Ztgsyl(NoTrans, ijb, n2, n1, a.Off(i-1, i-1), a, work.CMatrix(n2, opts), b.Off(i-1, i-1), b, work.CMatrixOff(n1*n2, n2, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
						panic(err)
					}
				} else {
					//                 Solve the transposed variant.
					if dscale, *dif.GetPtr(1), ierr, err = Ztgsyl(ConjTrans, ijb, n2, n1, a.Off(i-1, i-1), a, work.CMatrix(n2, opts), b, b.Off(i-1, i-1), work.CMatrixOff(n1*n2, n2, opts), work.Off(n1*n2*2), lwork-2*n1*n2, iwork); err != nil {
						panic(err)
					}
				}
				goto label50
			}
			dif.Set(1, dscale/dif.Get(1))
		}
	}

	//     If B(K,K) is complex, make it real and positive (normalization
	//     of the generalized Schur form) and Store the generalized
	//     eigenvalues of reordered pair (A, B)
	for k = 1; k <= n; k++ {
		dscale = b.GetMag(k-1, k-1)
		if dscale > safmin {
			temp1 = b.GetConj(k-1, k-1) / complex(dscale, 0)
			temp2 = b.Get(k-1, k-1) / complex(dscale, 0)
			b.SetRe(k-1, k-1, dscale)
			goblas.Zscal(n-k, temp1, b.CVector(k-1, k))
			goblas.Zscal(n-k+1, temp1, a.CVector(k-1, k-1))
			if wantq {
				goblas.Zscal(n, temp2, q.CVector(0, k-1, 1))
			}
		} else {
			b.SetRe(k-1, k-1, zero)
		}

		alpha.Set(k-1, a.Get(k-1, k-1))
		beta.Set(k-1, b.Get(k-1, k-1))

	}

label70:
	;

	work.SetRe(0, float64(lwmin))
	(*iwork)[0] = liwmin

	return
}
