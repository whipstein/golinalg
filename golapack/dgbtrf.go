package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgbtrf computes an LU factorization of a real m-by-n band matrix A
// using partial pivoting with row interchanges.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Dgbtrf(m, n, kl, ku int, ab *mat.Matrix, ipiv *[]int) (info int, err error) {
	var one, temp, zero float64
	var i, i2, i3, ii, ip, j, j2, j3, jb, jj, jm, jp, ju, k2, km, kv, ldwork, nb, nbmax, nw int

	one = 1.0
	zero = 0.0
	nbmax = 64
	ldwork = nbmax + 1
	work13 := mf(ldwork, nbmax, opts)
	work31 := mf(ldwork, nbmax, opts)

	//     KV is the number of superdiagonals in the factor U, allowing for
	//     fill-in
	kv = ku + kl

	//     Test the input parameters.
	info = 0
	if m < 0 {
		info = -1
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		info = -2
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 {
		info = -3
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		info = -4
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if ab.Rows < kl+kv+1 {
		info = -6
		err = fmt.Errorf("ab.Rows < kl+kv+1: ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	}
	if err != nil {
		gltest.Xerbla2("Dgbtrf", err)
		return
	}

	//     Quick return if possible
	if m == 0 || n == 0 {
		return
	}

	//     Determine the block size for this environment
	nb = Ilaenv(1, "Dgbtrf", []byte{' '}, m, n, kl, ku)

	//     The block size must not exceed the limit set by the size of the
	//     local arrays WORK13 and WORK31.
	nb = min(nb, nbmax)

	if nb <= 1 || nb > kl {
		//        Use unblocked code
		if info, err = Dgbtf2(m, n, kl, ku, ab, ipiv); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		//
		//        Zero the superdiagonal elements of the work array WORK13
		for j = 1; j <= nb; j++ {
			for i = 1; i <= j-1; i++ {
				work13.Set(i-1, j-1, zero)
			}
		}

		//        Zero the subdiagonal elements of the work array WORK31
		for j = 1; j <= nb; j++ {
			for i = j + 1; i <= nb; i++ {
				work31.Set(i-1, j-1, zero)
			}
		}

		//        Gaussian elimination with partial pivoting
		//
		//        Set fill-in elements in columns KU+2 to KV to zero
		for j = ku + 2; j <= min(kv, n); j++ {
			for i = kv - j + 2; i <= kl; i++ {
				ab.Set(i-1, j-1, zero)
			}
		}

		//        JU is the index of the last column affected by the current
		//        stage of the factorization
		ju = 1

		for j = 1; j <= min(m, n); j += nb {
			jb = min(nb, min(m, n)-j+1)
			//           The active part of the matrix is partitioned
			//
			//              A11   A12   A13
			//              A21   A22   A23
			//              A31   A32   A33
			//
			//           Here A11, A21 and A31 denote the current block of JB columns
			//           which is about to be factorized. The number of rows in the
			//           partitioning are JB, I2, I3 respectively, and the numbers
			//           of columns are JB, J2, J3. The superdiagonal elements of A13
			//           and the subdiagonal elements of A31 lie outside the band.
			i2 = min(kl-jb, m-j-jb+1)
			i3 = min(jb, m-j-kl+1)

			//           J2 and J3 are computed after JU has been updated.
			//
			//           Factorize the current block of JB columns
			for jj = j; jj <= j+jb-1; jj++ {
				//              Set fill-in elements in column JJ+KV to zero
				if jj+kv <= n {
					for i = 1; i <= kl; i++ {
						ab.Set(i-1, jj+kv-1, zero)
					}
				}

				//              Find pivot and test for singularity. KM is the number of
				//              subdiagonal elements in the current column.
				km = min(kl, m-jj)
				jp = goblas.Idamax(km+1, ab.Vector(kv, jj-1, 1))
				(*ipiv)[jj-1] = jp + jj - j
				if ab.Get(kv+jp-1, jj-1) != zero {
					ju = max(ju, min(jj+ku+jp-1, n))
					if jp != 1 {
						//                    Apply interchange to columns J to J+JB-1
						if jp+jj-1 < j+kl {
							goblas.Dswap(jb, ab.Vector(kv+1+jj-j-1, j-1), ab.Vector(kv+jp+jj-j-1, j-1))
						} else {
							//                       The interchange affects columns J to JJ-1 of A31
							//                       which are stored in the work array WORK31
							goblas.Dswap(jj-j, ab.Vector(kv+1+jj-j-1, j-1), work31.Vector(jp+jj-j-kl-1, 0))
							goblas.Dswap(j+jb-jj, ab.Vector(kv, jj-1), ab.Vector(kv+jp-1, jj-1))
						}
					}

					//                 Compute multipliers
					goblas.Dscal(km, one/ab.Get(kv, jj-1), ab.Vector(kv+2-1, jj-1, 1))

					//                 Update trailing submatrix within the band and within
					//                 the current block. JM is the index of the last column
					//                 which needs to be updated.
					jm = min(ju, j+jb-1)
					if jm > jj {
						err = goblas.Dger(km, jm-jj, -one, ab.Vector(kv+2-1, jj-1, 1), ab.Vector(kv-1, jj), ab.Off(kv, jj))
					}
				} else {
					//                 If pivot is zero, set INFO to the index of the pivot
					//                 unless a zero pivot has already been found.
					if info == 0 {
						info = jj
					}
				}

				//              Copy current column of A31 into the work array WORK31
				nw = min(jj-j+1, i3)
				if nw > 0 {
					goblas.Dcopy(nw, ab.Vector(kv+kl+1-jj+j-1, jj-1, 1), work31.Vector(0, jj-j, 1))
				}
			}
			if j+jb <= n {
				//              Apply the row interchanges to the other blocks.
				j2 = min(ju-j+1, kv) - jb
				j3 = max(0, ju-j-kv+1)

				//              Use DLASWP to apply the row interchanges to A12, A22, and
				//              A32.
				Dlaswp(j2, ab.Off(kv+1-jb-1, j+jb-1), 1, jb, *toSlice(ipiv, j-1), 1)

				//              Adjust the pivot indices.
				for i = j; i <= j+jb-1; i++ {
					(*ipiv)[i-1] = (*ipiv)[i-1] + j - 1
				}

				//              Apply the row interchanges to A13, A23, and A33
				//              columnwise.
				k2 = j - 1 + jb + j2
				for i = 1; i <= j3; i++ {
					jj = k2 + i
					for ii = j + i - 1; ii <= j+jb-1; ii++ {
						ip = (*ipiv)[ii-1]
						if ip != ii {
							temp = ab.Get(kv+1+ii-jj-1, jj-1)
							ab.Set(kv+1+ii-jj-1, jj-1, ab.Get(kv+1+ip-jj-1, jj-1))
							ab.Set(kv+1+ip-jj-1, jj-1, temp)
						}
					}
				}

				//              Update the relevant part of the trailing submatrix
				if j2 > 0 {
					//                 Update A12
					err = goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, jb, j2, one, ab.Off(kv, j-1), ab.Off(kv+1-jb-1, j+jb-1))

					if i2 > 0 {
						//                    Update A22
						err = goblas.Dgemm(mat.NoTrans, mat.NoTrans, i2, j2, jb, -one, ab.Off(kv+1+jb-1, j-1), ab.Off(kv+1-jb-1, j+jb-1), one, ab.Off(kv, j+jb-1))
					}

					if i3 > 0 {
						//                    Update A32
						err = goblas.Dgemm(mat.NoTrans, mat.NoTrans, i3, j2, jb, -one, work31, ab.Off(kv+1-jb-1, j+jb-1), one, ab.Off(kv+kl+1-jb-1, j+jb-1))
					}
				}

				if j3 > 0 {
					//                 Copy the lower triangle of A13 into the work array
					//                 WORK13
					for jj = 1; jj <= j3; jj++ {
						for ii = jj; ii <= jb; ii++ {
							work13.Set(ii-1, jj-1, ab.Get(ii-jj, jj+j+kv-1-1))
						}
					}

					//                 Update A13 in the work array
					err = goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, jb, j3, one, ab.Off(kv, j-1), work13)

					if i2 > 0 {
						//                    Update A23
						err = goblas.Dgemm(mat.NoTrans, mat.NoTrans, i2, j3, jb, -one, ab.Off(kv+1+jb-1, j-1), work13, one, ab.Off(1+jb-1, j+kv-1))
					}

					if i3 > 0 {
						//                    Update A33
						err = goblas.Dgemm(mat.NoTrans, mat.NoTrans, i3, j3, jb, -one, work31, work13, one, ab.Off(1+kl-1, j+kv-1))
					}

					//                 Copy the lower triangle of A13 back into place
					for jj = 1; jj <= j3; jj++ {
						for ii = jj; ii <= jb; ii++ {
							ab.Set(ii-jj, jj+j+kv-1-1, work13.Get(ii-1, jj-1))
						}
					}
				}
			} else {
				//              Adjust the pivot indices.
				for i = j; i <= j+jb-1; i++ {
					(*ipiv)[i-1] = (*ipiv)[i-1] + j - 1
				}
			}

			//           Partially undo the interchanges in the current block to
			//           restore the upper triangular form of A31 and copy the upper
			//           triangle of A31 back into place
			for jj = j + jb - 1; jj >= j; jj-- {
				jp = (*ipiv)[jj-1] - jj + 1
				if jp != 1 {
					//                 Apply interchange to columns J to JJ-1
					if jp+jj-1 < j+kl {
						//                    The interchange does not affect A31
						goblas.Dswap(jj-j, ab.Vector(kv+1+jj-j-1, j-1), ab.Vector(kv+jp+jj-j-1, j-1))
					} else {
						//                    The interchange does affect A31
						goblas.Dswap(jj-j, ab.Vector(kv+1+jj-j-1, j-1), work31.Vector(jp+jj-j-kl-1, 0))
					}
				}

				//              Copy the current column of A31 back into place
				nw = min(i3, jj-j+1)
				if nw > 0 {
					goblas.Dcopy(nw, work31.Vector(0, jj-j, 1), ab.Vector(kv+kl+1-jj+j-1, jj-1, 1))
				}
			}
		}
	}

	return
}
